#include "llama.h"

// Hook used by the Android wrapper to invalidate token/cache bookkeeping whenever
// the native model/context is replaced or unloaded. KV reuse is intentionally disabled
// for this mobile runtime, so there is no separate Android-side cache to invalidate.
static void clear_android_generation_cache() {}
#include <signal.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>

// Keep the JNI implementation in native_runtime.cpp. Its own fatal handler is
// retained under a private name; the Android-specific handler below is installed
// explicitly after nativeInit() returns, so it cannot be accidentally shadowed.
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

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeInstallFatalHandlers(
        JNIEnv *, jclass) {
    install_native_fatal_handlers();
}


static bool emit_complete_utf8(JNIEnv * env, jobject listener, jmethodID on_token, std::string & pending) {
    std::u16string utf16;
    size_t i = 0;

    while (i < pending.size()) {
        const unsigned char c0 = static_cast<unsigned char>(pending[i]);
        uint32_t codepoint = 0;
        size_t width = 0;

        if (c0 <= 0x7F) {
            codepoint = c0;
            width = 1;
        } else if (c0 >= 0xC2 && c0 <= 0xDF) {
            if (i + 1 >= pending.size()) break;
            const unsigned char c1 = static_cast<unsigned char>(pending[i + 1]);
            if ((c1 & 0xC0) != 0x80) {
                codepoint = 0xFFFD;
                width = 1;
            } else {
                codepoint = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
                width = 2;
            }
        } else if (c0 >= 0xE0 && c0 <= 0xEF) {
            if (i + 2 >= pending.size()) break;
            const unsigned char c1 = static_cast<unsigned char>(pending[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(pending[i + 2]);
            const bool valid =
                    (c1 & 0xC0) == 0x80 &&
                    (c2 & 0xC0) == 0x80 &&
                    !(c0 == 0xE0 && c1 < 0xA0) &&
                    !(c0 == 0xED && c1 >= 0xA0);
            if (!valid) {
                codepoint = 0xFFFD;
                width = 1;
            } else {
                codepoint = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
                width = 3;
            }
        } else if (c0 >= 0xF0 && c0 <= 0xF4) {
            if (i + 3 >= pending.size()) break;
            const unsigned char c1 = static_cast<unsigned char>(pending[i + 1]);
            const unsigned char c2 = static_cast<unsigned char>(pending[i + 2]);
            const unsigned char c3 = static_cast<unsigned char>(pending[i + 3]);
            const bool valid =
                    (c1 & 0xC0) == 0x80 &&
                    (c2 & 0xC0) == 0x80 &&
                    (c3 & 0xC0) == 0x80 &&
                    !(c0 == 0xF0 && c1 < 0x90) &&
                    !(c0 == 0xF4 && c1 >= 0x90);
            if (!valid) {
                codepoint = 0xFFFD;
                width = 1;
            } else {
                codepoint = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) |
                            ((c2 & 0x3F) << 6) | (c3 & 0x3F);
                width = 4;
            }
        } else {
            codepoint = 0xFFFD;
            width = 1;
        }

        if (codepoint <= 0xFFFF) {
            utf16.push_back(static_cast<char16_t>(codepoint));
        } else {
            codepoint -= 0x10000;
            utf16.push_back(static_cast<char16_t>(0xD800 + (codepoint >> 10)));
            utf16.push_back(static_cast<char16_t>(0xDC00 + (codepoint & 0x3FF)));
        }
        i += width;
    }

    if (i == 0) return true;

    jstring chunk = env->NewString(reinterpret_cast<const jchar *>(utf16.data()),
                                   static_cast<jsize>(utf16.size()));
    if (!chunk) return false;

    env->CallVoidMethod(listener, on_token, chunk);
    env->DeleteLocalRef(chunk);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        return false;
    }

    pending.erase(0, i);
    return true;
}

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
    if (!g_context || !g_model) { checkpoint("NATIVE_GENERATE_NO_MODEL"); return 2; }
    if (!jprompt || !listener) { checkpoint("NATIVE_GENERATE_INVALID_ARGUMENT"); return 3; }
    const char * prompt = env->GetStringUTFChars(jprompt, nullptr);
    if (!prompt) { checkpoint("NATIVE_GENERATE_PROMPT_UTF8_FAILED"); return 3; }
    const std::string prompt_text(prompt);
    env->ReleaseStringUTFChars(jprompt, prompt);
    g_stop.store(false, std::memory_order_relaxed);

    // Keep the resident llama context and OpenCL graph allocations alive between
    // turns. Recreating the context here adds avoidable latency and repeatedly
    // allocates/frees GPU resources. Start every turn from an empty llama memory
    // instead; this clears the KV/recurrent state without changing model weights,
    // sampling parameters, GPU offload, or context capacity.
    if (!g_model || !g_context) {
        checkpoint("NATIVE_GENERATE_CONTEXT_UNAVAILABLE");
        return 2;
    }
    checkpoint("NATIVE_GENERATE_MEMORY_CLEAR_STARTED");
    llama_memory_t memory = llama_get_memory(g_context);
    if (!memory) {
        checkpoint("NATIVE_GENERATE_MEMORY_UNAVAILABLE");
        return 2;
    }
    llama_memory_clear(memory, true);
    checkpoint("NATIVE_GENERATE_MEMORY_CLEAR_COMPLETED");

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) { checkpoint("NATIVE_GENERATE_VOCAB_MISSING"); return 4; }
    const int n_prompt = -llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) { checkpoint("NATIVE_GENERATE_TOKENIZE_COUNT_FAILED"); return 5; }
    std::vector<llama_token> prompt_tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) { checkpoint("NATIVE_GENERATE_TOKENIZE_FAILED"); return 5; }
    const int n_ctx = (int) llama_n_ctx(g_context);
    if (n_prompt >= n_ctx) { append_native_trace((std::string("NATIVE_GENERATE_CONTEXT_OVERFLOW promptTokens=") + std::to_string(n_prompt) + " context=" + std::to_string(n_ctx)).c_str()); checkpoint("NATIVE_GENERATE_CONTEXT_OVERFLOW"); return 6; }
    const int requested_max = std::max(1, (int) max_tokens);
    const int max_predict = std::min(requested_max, n_ctx - n_prompt);
    if (max_predict <= 0) { checkpoint("NATIVE_GENERATE_NO_TOKEN_CAPACITY"); return 6; }
    auto sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (!sampler) { checkpoint("NATIVE_GENERATE_SAMPLER_INIT_FAILED"); return 7; }
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
    append_native_trace((std::string("NATIVE_GENERATE_PARAMS promptTokens=") + std::to_string(n_prompt) + " maxTokens=" + std::to_string(max_predict) + " temperature=" + std::to_string(temp) + " topK=" + std::to_string(k) + " topP=" + std::to_string(p) + " minP=" + std::to_string(mp)).c_str());
    int generated = 0;
    int position = 0;
    std::string pending_utf8;
    int result = 0;
    const auto generation_started_at = std::chrono::steady_clock::now();
    bool prefill_recorded = false;
    while (position + batch.n_tokens < n_prompt + max_predict) {
        if (g_stop.load(std::memory_order_relaxed)) { result = 9; checkpoint("NATIVE_GENERATE_STOPPED"); break; }
        const int decode_result = llama_decode(g_context, batch);
        if (!prefill_recorded && decode_result == 0) {
            const auto now = std::chrono::steady_clock::now();
            const auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - generation_started_at).count();
            append_native_trace((std::string("NATIVE_PREFILL_COMPLETED promptTokens=") + std::to_string(n_prompt) + " reusedTokens=" + std::to_string(0) + " prefillMs=" + std::to_string(prefill_ms)).c_str());
            prefill_recorded = true;
        }
        if (decode_result != 0) { append_native_trace((std::string("NATIVE_GENERATE_DECODE_FAILED code=") + std::to_string(decode_result)).c_str()); checkpoint("NATIVE_GENERATE_DECODE_FAILED"); result = 10; break; }
        position += batch.n_tokens;
        const llama_token token = llama_sampler_sample(sampler, g_context, -1);
        if (llama_vocab_is_eog(vocab, token)) { checkpoint("NATIVE_GENERATE_EOG"); break; }
        char piece[1024];
        int piece_size = llama_token_to_piece(vocab, token, piece, (int) sizeof(piece), 0, true);
        if (piece_size < 0) { append_native_trace((std::string("NATIVE_GENERATE_TOKEN_TO_PIECE_FAILED size=") + std::to_string(piece_size)).c_str()); checkpoint("NATIVE_GENERATE_TOKEN_TO_PIECE_FAILED"); result = 11; break; }
        if (piece_size > 0) {
            // llama_token_to_piece() deliberately does not null-terminate its output.
            // Consume exactly piece_size bytes and keep incomplete UTF-8 sequences
            // until the following token supplies the remaining bytes.
            pending_utf8.append(piece, static_cast<size_t>(piece_size));
            if (!emit_complete_utf8(env, listener, on_token, pending_utf8)) {
                checkpoint("NATIVE_GENERATE_JSTRING_FAILED");
                result = 12;
                break;
            }
        }
        ++generated;
        batch = llama_batch_get_one(const_cast<llama_token *>(&token), 1);
    }
    if (!pending_utf8.empty() && result == 0) {
        // A valid model output should not leave a partial UTF-8 sequence behind.
        // Emit it as U+FFFD rather than passing unterminated bytes to JNI.
        pending_utf8.clear();
        const jchar replacement = 0xFFFD;
        jstring chunk = env->NewString(&replacement, 1);
        if (chunk) {
            env->CallVoidMethod(listener, on_token, chunk);
            env->DeleteLocalRef(chunk);
            if (env->ExceptionCheck()) env->ExceptionClear();
        }
    }
    env->DeleteLocalRef(listener_class);
    llama_sampler_free(sampler);
    const auto generation_finished_at = std::chrono::steady_clock::now();
    const auto generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(generation_finished_at - generation_started_at).count();
    const double decode_tokens_per_sec = generated > 0 && generation_ms > 0 ? (1000.0 * static_cast<double>(generated) / static_cast<double>(generation_ms)) : 0.0;
    append_native_trace((std::string("NATIVE_GENERATE_COMPLETED generatedTokens=") + std::to_string(generated) + " result=" + std::to_string(result) + " generationMs=" + std::to_string(generation_ms) + " decodeTokensPerSec=" + std::to_string(decode_tokens_per_sec)).c_str());
    checkpoint(result == 0 ? "NATIVE_GENERATE_RETURNED_SUCCESS" : "NATIVE_GENERATE_RETURNED_FAILURE");
    return result;
}
