#include "llama.h"
#include <vector>

// Tokens represented by the resident prompt prefix in the native KV cache.
static std::vector<llama_token> g_cached_prompt_tokens;
// Hybrid/recurrent models cannot safely trim their recurrent state with
// llama_memory_seq_rm(). Keep an exact post-prefill sequence snapshot instead.
static std::vector<uint8_t> g_cached_prompt_state;
static void clear_android_generation_cache() {
    g_cached_prompt_tokens.clear();
    g_cached_prompt_state.clear();
}

#include <signal.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>

static std::mutex g_native_runtime_mutex;

// Keep the JNI implementation in native_runtime.cpp. Its own fatal handler is
// retained under a private name; the Android-specific handler below is installed
// explicitly after nativeInit() returns, so it cannot be accidentally shadowed.
#define install_native_fatal_handlers install_native_fatal_handlers_legacy
#include "native_runtime.cpp"
#undef install_native_fatal_handlers

static bool model_uses_recurrent_memory() {
    return g_model && (llama_model_is_recurrent(g_model) || llama_model_is_hybrid(g_model));
}

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
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeCountTokens(
        JNIEnv * env,
        jclass,
        jstring jprompt) {
    std::lock_guard<std::mutex> runtime_lock(g_native_runtime_mutex);
    if (!g_model || !jprompt) return -1;
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) return -1;
    const char * prompt = env->GetStringUTFChars(jprompt, nullptr);
    if (!prompt) return -1;
    const size_t text_size = std::strlen(prompt);
    if (text_size > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        env->ReleaseStringUTFChars(jprompt, prompt);
        return -2;
    }
    const int32_t text_len = static_cast<int32_t>(text_size);
    const int count = -llama_tokenize(vocab, prompt, text_len, nullptr, 0, true, true);
    env->ReleaseStringUTFChars(jprompt, prompt);
    return count >= 0 ? count : -2;
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
    std::lock_guard<std::mutex> runtime_lock(g_native_runtime_mutex);
    checkpoint("NATIVE_GENERATE_ENTERED");
    if (!g_context || !g_model) { checkpoint("NATIVE_GENERATE_NO_MODEL"); return 2; }
    if (!jprompt || !listener) { checkpoint("NATIVE_GENERATE_INVALID_ARGUMENT"); return 3; }
    const char * prompt = env->GetStringUTFChars(jprompt, nullptr);
    if (!prompt) { checkpoint("NATIVE_GENERATE_PROMPT_UTF8_FAILED"); return 3; }
    const std::string prompt_text(prompt);
    env->ReleaseStringUTFChars(jprompt, prompt);
    g_stop.store(false, std::memory_order_relaxed);

    if (!g_model || !g_context) {
        checkpoint("NATIVE_GENERATE_CONTEXT_UNAVAILABLE");
        return 2;
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!vocab) { checkpoint("NATIVE_GENERATE_VOCAB_MISSING"); return 4; }
    const int n_prompt = -llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), nullptr, 0, true, true);
    if (n_prompt <= 0) { checkpoint("NATIVE_GENERATE_TOKENIZE_COUNT_FAILED"); return 5; }
    std::vector<llama_token> prompt_tokens((size_t) n_prompt);
    if (llama_tokenize(vocab, prompt_text.c_str(), prompt_text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) { checkpoint("NATIVE_GENERATE_TOKENIZE_FAILED"); return 5; }
    const int n_ctx = (int) llama_n_ctx(g_context);
    append_native_trace((std::string("NATIVE_PROMPT_TOKENIZED promptTokens=") + std::to_string(n_prompt) + " context=" + std::to_string(n_ctx)).c_str());
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
        // Keep the domain setting effective: the Kotlin runtime exposes a
        // repeat penalty of 1.1 by default, but the native path previously
        // ignored it entirely. Apply it before top-k/top-p so the candidate
        // set is shaped once and sampling remains bounded.
        constexpr float kRepeatPenalty = 1.1f;
        constexpr int kRepeatLastN = 64;
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocab), kRepeatLastN, kRepeatPenalty, 0.0f, 0.0f));
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
    llama_memory_t memory = llama_get_memory(g_context);
    if (!memory) {
        checkpoint("NATIVE_GENERATE_MEMORY_UNAVAILABLE");
        return 8;
    }

    size_t common_prefix = 0;
    while (common_prefix < g_cached_prompt_tokens.size() &&
           common_prefix < prompt_tokens.size() &&
           g_cached_prompt_tokens[common_prefix] == prompt_tokens[common_prefix]) {
        ++common_prefix;
    }

    const bool hybrid_memory = model_uses_recurrent_memory();
    // Attention-only models can trim the resident KV cache by sequence position.
    // Hybrid/recurrent models cannot: their recurrent state is path-dependent.
    // For those models we reuse only an exact cached prompt state captured
    // immediately after prompt prefill and before generation.
    const bool cache_supported = !llama_model_has_encoder(g_model);
    bool cache_reused = cache_supported && !g_cached_prompt_tokens.empty() && common_prefix > 0;
    size_t reuse_prefix = 0;
    const bool exact_cached_prefix = !g_cached_prompt_tokens.empty() &&
            common_prefix == g_cached_prompt_tokens.size();
    append_native_trace((std::string("NATIVE_KV_CACHE_PREFIX_CHECK supported=") +
        (cache_supported ? "1" : "0") +
        " hybrid=" + (hybrid_memory ? "1" : "0") +
        " cachedTokens=" + std::to_string(g_cached_prompt_tokens.size()) +
        " commonPrefix=" + std::to_string(common_prefix) +
        " stateBytes=" + std::to_string(g_cached_prompt_state.size())).c_str());

    if (hybrid_memory) {
        // A hybrid state is safe to reuse only when the entire cached prompt is
        // still an exact prefix. Partial trimming of recurrent state is invalid.
        if (exact_cached_prefix && !g_cached_prompt_state.empty()) {
            llama_memory_clear(memory, true);
            const size_t restored = llama_state_seq_set_data(
                    g_context,
                    g_cached_prompt_state.data(),
                    g_cached_prompt_state.size(),
                    0);
            if (restored > 0) {
                cache_reused = true;
                reuse_prefix = g_cached_prompt_tokens.size();
                append_native_trace((std::string("NATIVE_KV_CACHE_HYBRID_STATE_RESTORED tokens=") +
                    std::to_string(reuse_prefix) + " bytes=" +
                    std::to_string(restored)).c_str());
            } else {
                cache_reused = false;
                g_cached_prompt_state.clear();
                g_cached_prompt_tokens.clear();
                checkpoint("NATIVE_KV_CACHE_HYBRID_STATE_RESTORE_FAILED");
            }
        } else {
            cache_reused = false;
            g_cached_prompt_state.clear();
            g_cached_prompt_tokens.clear();
            llama_memory_clear(memory, true);
            checkpoint("NATIVE_KV_CACHE_HYBRID_REBUILD");
        }
    } else if (!cache_reused) {
        llama_memory_clear(memory, true);
        g_cached_prompt_tokens.clear();
        g_cached_prompt_state.clear();
        checkpoint("NATIVE_KV_CACHE_MISS_CLEARED");
    } else {
        const bool exact_prompt = common_prefix == prompt_tokens.size() &&
                                  common_prefix == g_cached_prompt_tokens.size();
        reuse_prefix = exact_prompt && common_prefix > 0 ? common_prefix - 1 : common_prefix;
        if (!llama_memory_seq_rm(memory, 0, (llama_pos) reuse_prefix, -1)) {
            llama_memory_clear(memory, true);
            g_cached_prompt_tokens.clear();
            g_cached_prompt_state.clear();
            reuse_prefix = 0;
            cache_reused = false;
            checkpoint("NATIVE_KV_CACHE_PARTIAL_REMOVE_UNSUPPORTED");
        } else {
            append_native_trace((std::string("NATIVE_KV_CACHE_HIT reusedTokens=") +
                std::to_string(reuse_prefix) + " promptTokens=" +
                std::to_string(prompt_tokens.size())).c_str());
        }
    }

    // Do not publish the new prompt as cacheable until its prefill has actually
    // completed. A failed decode must never advertise a partially-built state
    // as a valid prefix for the next turn.

    // The context is intentionally configured with a physical batch/ubatch of 128 on Android.
    // Never pass the entire 4K context as one llama_decode() batch: llama_decode() requires
    // callers to split larger prompt batches according to llama_n_batch(). Passing a 3K-4K
    // token prompt directly here was the long-prompt crash path and could also corrupt the
    // second-turn cache state on devices with strict OpenCL memory limits.
    const int decode_batch_size = std::max(1, (int) llama_n_batch(g_context));
    const int batch_capacity = decode_batch_size;
    llama_batch batch = llama_batch_init(batch_capacity, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        env->DeleteLocalRef(listener_class);
        llama_sampler_free(sampler);
        checkpoint("NATIVE_GENERATE_BATCH_INIT_FAILED");
        return 9;
    }
    auto fill_batch = [&](const llama_token *tokens, int count, int start_pos, bool output_last) {
        batch.n_tokens = count;
        for (int i = 0; i < count; ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = (llama_pos) (start_pos + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (output_last && i == count - 1) ? 1 : 0;
        }
    };

    append_native_trace((std::string("NATIVE_GENERATE_BATCH_CONFIG nBatch=") +
        std::to_string(decode_batch_size) + " promptTokens=" +
        std::to_string(n_prompt) + " reusedTokens=" +
        std::to_string(reuse_prefix)).c_str());

    int generated = 0;
    int position = (int) reuse_prefix;
    std::string pending_utf8;
    int result = 0;
    // JNI callbacks are expensive on Android. Keep native token generation
    // token-by-token, but deliver a small batch to Kotlin to reduce per-token
    // JNI/UTF-16 allocation overhead without changing model sampling.
    static constexpr int kStreamChunkTokens = 4;
    int tokens_since_emit = 0;
    const auto generation_started_at = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point prefill_finished_at = generation_started_at;
    bool prefill_recorded = false;

    // Prefill the new prompt in physical batches. Only the final prompt token
    // requests logits because that is the token used to sample the first output.
    append_native_trace((std::string("NATIVE_PREFILL_STARTED promptTokens=") + std::to_string(n_prompt) + " reusedTokens=" + std::to_string(reuse_prefix) + " remainingTokens=" + std::to_string(n_prompt - (int) reuse_prefix)).c_str());
    int prompt_offset = (int) reuse_prefix;
    while (prompt_offset < n_prompt) {
        if (g_stop.load(std::memory_order_relaxed)) { result = 9; checkpoint("NATIVE_GENERATE_STOPPED"); break; }
        const int count = std::min(decode_batch_size, n_prompt - prompt_offset);
        const bool final_prompt_batch = prompt_offset + count == n_prompt;
        fill_batch(prompt_tokens.data() + prompt_offset, count, prompt_offset, final_prompt_batch);
        const int decode_result = llama_decode(g_context, batch);
        if (decode_result != 0) {
            append_native_trace((std::string("NATIVE_GENERATE_PREFILL_FAILED code=") + std::to_string(decode_result) +
                " offset=" + std::to_string(prompt_offset) + " count=" + std::to_string(count)).c_str());
            g_cached_prompt_tokens.clear();
            llama_memory_clear(memory, true);
            checkpoint("NATIVE_GENERATE_PREFILL_FAILED_CACHE_RESET");
            result = 10;
            break;
        }
        position = prompt_offset + count;
        prompt_offset += count;
        if (final_prompt_batch) {
            const auto now = std::chrono::steady_clock::now();
            prefill_finished_at = now;
            const auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - generation_started_at).count();
            append_native_trace((std::string("NATIVE_PREFILL_COMPLETED promptTokens=") +
                std::to_string(n_prompt) + " reusedTokens=" + std::to_string(reuse_prefix) +
                " prefillMs=" + std::to_string(prefill_ms)).c_str());
            prefill_recorded = true;
        }
    }

    // Snapshot the post-prefill state before any sampled token mutates the
    // recurrent memory. This gives hybrid Qwen-family models an exact turn
    // checkpoint that can be restored on the next request.
    if (prefill_recorded && model_uses_recurrent_memory()) {
        static constexpr size_t kMaxHybridStateBytes = 128u * 1024u * 1024u;
        const size_t state_size = llama_state_seq_get_size(g_context, 0);
        if (state_size > 0 && state_size <= kMaxHybridStateBytes) {
            g_cached_prompt_state.resize(state_size);
            const size_t written = llama_state_seq_get_data(
                    g_context,
                    g_cached_prompt_state.data(),
                    g_cached_prompt_state.size(),
                    0);
            if (written > 0) {
                g_cached_prompt_state.resize(written);
                append_native_trace((std::string("NATIVE_KV_CACHE_HYBRID_STATE_CAPTURED tokens=") +
                    std::to_string(prompt_tokens.size()) + " bytes=" +
                    std::to_string(written)).c_str());
            } else {
                g_cached_prompt_state.clear();
                append_native_trace("NATIVE_KV_CACHE_HYBRID_STATE_CAPTURE_FAILED");
            }
        } else {
            g_cached_prompt_state.clear();
            append_native_trace((std::string("NATIVE_KV_CACHE_HYBRID_STATE_SKIPPED bytes=") +
                std::to_string(state_size) + " limit=" +
                std::to_string(kMaxHybridStateBytes)).c_str());
        }
    }

    if (result == 0) {
        while (generated < max_predict) {
            if (g_stop.load(std::memory_order_relaxed)) { result = 9; checkpoint("NATIVE_GENERATE_STOPPED"); break; }

            const llama_token token = llama_sampler_sample(sampler, g_context, -1);
            llama_sampler_accept(sampler, token);
            if (llama_vocab_is_eog(vocab, token)) { checkpoint("NATIVE_GENERATE_EOG"); break; }

            char piece[1024];
            int piece_size = llama_token_to_piece(vocab, token, piece, (int) sizeof(piece), 0, true);
            if (piece_size < 0) {
                append_native_trace((std::string("NATIVE_GENERATE_TOKEN_TO_PIECE_FAILED size=") + std::to_string(piece_size)).c_str());
                checkpoint("NATIVE_GENERATE_TOKEN_TO_PIECE_FAILED");
                result = 11;
                break;
            }
            if (piece_size > 0) {
                pending_utf8.append(piece, static_cast<size_t>(piece_size));
            }
            ++generated;
            ++tokens_since_emit;
            if (tokens_since_emit >= kStreamChunkTokens) {
                if (!emit_complete_utf8(env, listener, on_token, pending_utf8)) {
                    checkpoint("NATIVE_GENERATE_JSTRING_FAILED");
                    result = 12;
                    break;
                }
                tokens_since_emit = 0;
            }
            if (generated >= max_predict) break;

            // Decode exactly one sampled token. This keeps generation memory bounded
            // while retaining the full prompt/context and without changing sampling.
            fill_batch(&token, 1, position, true);
            const int decode_result = llama_decode(g_context, batch);
            if (decode_result != 0) {
                append_native_trace((std::string("NATIVE_GENERATE_DECODE_FAILED code=") + std::to_string(decode_result) +
                    " generated=" + std::to_string(generated)).c_str());
                g_cached_prompt_tokens.clear();
                llama_memory_clear(memory, true);
                checkpoint("NATIVE_GENERATE_DECODE_FAILED_CACHE_RESET");
                result = 10;
                break;
            }
            ++position;
        }
    }

    if (!pending_utf8.empty() && (result == 0 || result == 9)) {
        // Flush the final stream chunk before handling any partial UTF-8 sequence.
        if (!emit_complete_utf8(env, listener, on_token, pending_utf8)) {
            checkpoint("NATIVE_GENERATE_JSTRING_FAILED_FINAL_FLUSH");
            result = 12;
        }
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
    llama_batch_free(batch);
    env->DeleteLocalRef(listener_class);
    llama_sampler_free(sampler);
    const auto generation_finished_at = std::chrono::steady_clock::now();
    const auto generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(generation_finished_at - generation_started_at).count();
    const auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(generation_finished_at - prefill_finished_at).count();
    const double decode_tokens_per_sec = generated > 0 && decode_ms > 0 ? (1000.0 * static_cast<double>(generated) / static_cast<double>(decode_ms)) : 0.0;
    if (prefill_recorded && (result == 0 || result == 9)) {
        g_cached_prompt_tokens = prompt_tokens;
        if (!model_uses_recurrent_memory()) {
            g_cached_prompt_state.clear();
        }
        append_native_trace((std::string("NATIVE_KV_CACHE_PUBLISHED tokens=") + std::to_string(g_cached_prompt_tokens.size())).c_str());
    } else if (prefill_recorded) {
        g_cached_prompt_tokens.clear();
        g_cached_prompt_state.clear();
        llama_memory_clear(memory, true);
        checkpoint("NATIVE_KV_CACHE_NOT_PUBLISHED_AFTER_FAILED_GENERATION");
    }
    append_native_trace((std::string("NATIVE_GENERATE_COMPLETED generatedTokens=") + std::to_string(generated) +
        " result=" + std::to_string(result) +
        " generationMs=" + std::to_string(generation_ms) +
        " prefillMs=" + std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(prefill_finished_at - generation_started_at).count()) +
        " decodeMs=" + std::to_string(decode_ms) +
        " decodeTokensPerSec=" + std::to_string(decode_tokens_per_sec) +
        " streamChunkTokens=" + std::to_string(kStreamChunkTokens)).c_str());
    checkpoint(result == 0 ? "NATIVE_GENERATE_RETURNED_SUCCESS" : "NATIVE_GENERATE_RETURNED_FAILURE");
    return result;
}
