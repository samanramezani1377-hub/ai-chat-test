#include "llama.h"
#include <signal.h>
#include <unistd.h>
#include <vector>
#include <string>

// Keep the JNI implementation in native_runtime.cpp. Rename only its fatal
// handler installer while including it so Android gets the debuggerd-friendly
// handler below. Do NOT rename llama_model_load_from_file: the legacy wrapper
// previously did that and recursively called itself after macro expansion.
#define install_native_fatal_handlers install_native_fatal_handlers_legacy
#include "native_runtime.cpp"
#undef install_native_fatal_handlers

static void android_fatal_signal_handler(int signal_number, siginfo_t * info, void * raw_context) {
    if (g_native_fatal_fd >= 0) {
        const char signal_prefix[] = "NATIVE_FATAL_SIGNAL=";
        native_write_text(g_native_fatal_fd, signal_prefix, sizeof(signal_prefix) - 1);
        char value[24];
        int n = 0;
        int v = signal_number;
        if (v == 0) {
            value[n++] = '0';
        } else {
            char reverse[24];
            int r = 0;
            while (v > 0 && r < (int) sizeof(reverse)) {
                reverse[r++] = (char) ('0' + (v % 10));
                v /= 10;
            }
            while (r > 0) value[n++] = reverse[--r];
        }
        native_write_text(g_native_fatal_fd, value, (size_t) n);
        native_write_text(g_native_fatal_fd, "\n", 1);

        uintptr_t pc = 0;
        uintptr_t lr = 0;
#if defined(__aarch64__)
        if (raw_context) {
            const ucontext_t * context = static_cast<const ucontext_t *>(raw_context);
            pc = (uintptr_t) context->uc_mcontext.pc;
            lr = (uintptr_t) context->uc_mcontext.regs[30];
        }
#endif
        native_write_hex(g_native_fatal_fd, "NATIVE_FATAL_PC=", pc);
        native_write_hex(g_native_fatal_fd, "NATIVE_FATAL_LR=", lr);
        native_write_hex(g_native_fatal_fd, "NATIVE_FATAL_FAULT_ADDR=", info ? (uintptr_t) info->si_addr : 0);
        fsync(g_native_fatal_fd);
    }

    // Restore the default disposition and re-raise on the crashing thread so
    // Android debuggerd can generate the native tombstone/backtrace.
    struct sigaction default_action{};
    sigemptyset(&default_action.sa_mask);
    default_action.sa_handler = SIG_DFL;
    sigaction(signal_number, &default_action, nullptr);
    raise(signal_number);
    _exit(128 + signal_number);
}

static void install_native_fatal_handlers() {
    struct sigaction action{};
    sigemptyset(&action.sa_mask);
    action.sa_sigaction = android_fatal_signal_handler;
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
    sigaction(SIGABRT, &action, nullptr);
    sigaction(SIGILL, &action, nullptr);
    sigaction(SIGFPE, &action, nullptr);
}

// Native generation bridge. This lives in the same translation unit as
// native_runtime.cpp so it can use the runtime-owned model/context safely.
// Kotlin calls this exact JNI symbol from NativeLlamaCpp.generate().
extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeGenerate(
        JNIEnv * env,
        jclass,
        jstring jprompt,
        jint max_tokens,
        jfloat temperature,
        jint top_k,
        jfloat top_p,
        jfloat min_p,
        jobject listener) {
    checkpoint("NATIVE_GENERATE_ENTERED");

    if (!g_context || !g_model) {
        checkpoint("NATIVE_GENERATE_NO_MODEL");
        return 2;
    }
    if (!jprompt || !listener) {
        checkpoint("NATIVE_GENERATE_INVALID_ARGUMENT");
        return 3;
    }

    const char * prompt = env->GetStringUTFChars(jprompt, nullptr);
    if (!prompt) {
        checkpoint("NATIVE_GENERATE_PROMPT_UTF8_FAILED");
        return 3;
    }
    const std::string prompt_text(prompt);
    env->ReleaseStringUTFChars(jprompt, prompt);

    g_stop.store(false, std::memory_order_relaxed);
    llama_memory_clear(llama_get_memory(g_context), true);

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) {
        checkpoint("NATIVE_GENERATE_VOCAB_MISSING");
        return 4;
    }

    const int n_prompt = -llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) {
        checkpoint("NATIVE_GENERATE_TOKENIZE_COUNT_FAILED");
        return 5;
    }

    std::vector<llama_token> prompt_tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        checkpoint("NATIVE_GENERATE_TOKENIZE_FAILED");
        return 5;
    }

    const int n_ctx = (int) llama_n_ctx(g_context);
    if (n_prompt >= n_ctx) {
        append_native_trace((std::string("NATIVE_GENERATE_CONTEXT_OVERFLOW promptTokens=") +
            std::to_string(n_prompt) + " context=" + std::to_string(n_ctx)).c_str());
        checkpoint("NATIVE_GENERATE_CONTEXT_OVERFLOW");
        return 6;
    }

    const int requested_max = std::max(1, (int) max_tokens);
    const int max_predict = std::min(requested_max, n_ctx - n_prompt);
    if (max_predict <= 0) {
        checkpoint("NATIVE_GENERATE_NO_TOKEN_CAPACITY");
        return 6;
    }

    auto sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (!sampler) {
        checkpoint("NATIVE_GENERATE_SAMPLER_INIT_FAILED");
        return 7;
    }

    const float temp = std::max(0.0f, (float) temperature);
    const int k = std::max(0, (int) top_k);
    const float p = std::clamp((float) top_p, 0.0f, 1.0f);
    const float mp = std::clamp((float) min_p, 0.0f, 1.0f);

    if (temp <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        if (k > 0) llama_sampler_chain_add(sampler, llama_sampler_init_top_k(k));
        if (p > 0.0f && p < 1.0f) llama_sampler_chain_add(sampler, llama_sampler_init_top_p(p, 1));
        if (mp > 0.0f) llama_sampler_chain_add(sampler, llama_sampler_init_min_p(mp, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    jclass listener_class = env->GetObjectClass(listener);
    jmethodID on_token = listener_class ? env->GetMethodID(listener_class, "onToken", "(Ljava/lang/String;)V") : nullptr;
    if (!on_token) {
        if (listener_class) env->DeleteLocalRef(listener_class);
        llama_sampler_free(sampler);
        checkpoint("NATIVE_GENERATE_LISTENER_METHOD_MISSING");
        return 8;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
    if (llama_model_has_encoder(g_model)) {
        if (llama_encode(g_context, batch)) {
            env->DeleteLocalRef(listener_class);
            llama_sampler_free(sampler);
            checkpoint("NATIVE_GENERATE_ENCODE_FAILED");
            return 9;
        }
        llama_token decoder_start = llama_model_decoder_start_token(g_model);
        if (decoder_start == LLAMA_TOKEN_NULL) decoder_start = llama_vocab_bos(vocab);
        batch = llama_batch_get_one(&decoder_start, 1);
    }

    append_native_trace((std::string("NATIVE_GENERATE_PARAMS promptTokens=") + std::to_string(n_prompt) +
        " maxTokens=" + std::to_string(max_predict) + " temperature=" + std::to_string(temp) +
        " topK=" + std::to_string(k) + " topP=" + std::to_string(p) + " minP=" + std::to_string(mp)).c_str());

    int generated = 0;
    int position = 0;
    int result = 0;
    while (position + batch.n_tokens < n_prompt + max_predict) {
        if (g_stop.load(std::memory_order_relaxed)) {
            result = 9;
            checkpoint("NATIVE_GENERATE_STOPPED");
            break;
        }

        const int decode_result = llama_decode(g_context, batch);
        if (decode_result != 0) {
            append_native_trace((std::string("NATIVE_GENERATE_DECODE_FAILED code=") + std::to_string(decode_result)).c_str());
            checkpoint("NATIVE_GENERATE_DECODE_FAILED");
            result = 10;
            break;
        }
        position += batch.n_tokens;

        const llama_token token = llama_sampler_sample(sampler, g_context, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            checkpoint("NATIVE_GENERATE_EOG");
            break;
        }

        char piece[1024];
        int piece_size = llama_token_to_piece(vocab, token, piece, (int) sizeof(piece), 0, true);
        if (piece_size < 0) {
            append_native_trace((std::string("NATIVE_GENERATE_TOKEN_TO_PIECE_FAILED size=") + std::to_string(piece_size)).c_str());
            checkpoint("NATIVE_GENERATE_TOKEN_TO_PIECE_FAILED");
            result = 11;
            break;
        }

        if (piece_size > 0) {
            jstring chunk = env->NewStringUTF(piece);
            if (!chunk) {
                checkpoint("NATIVE_GENERATE_JSTRING_FAILED");
                result = 12;
                break;
            }
            env->CallVoidMethod(listener, on_token, chunk);
            env->DeleteLocalRef(chunk);
            if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
                checkpoint("NATIVE_GENERATE_LISTENER_EXCEPTION");
                result = 13;
                break;
            }
        }

        ++generated;
        batch = llama_batch_get_one(const_cast<llama_token *>(&token), 1);
    }

    env->DeleteLocalRef(listener_class);
    llama_sampler_free(sampler);

    append_native_trace((std::string("NATIVE_GENERATE_COMPLETED generatedTokens=") + std::to_string(generated) +
        " result=" + std::to_string(result)).c_str());
    checkpoint(result == 0 ? "NATIVE_GENERATE_RETURNED_SUCCESS" : "NATIVE_GENERATE_RETURNED_FAILURE");
    return result;
}
