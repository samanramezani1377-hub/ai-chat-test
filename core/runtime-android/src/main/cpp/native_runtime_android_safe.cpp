#include "llama.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <sys/stat.h>
#include <fstream>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <cstring>

// Tokens represented by the resident prompt prefix in the native KV cache.
static std::vector<llama_token> g_cached_prompt_tokens;
// A stable system-only sequence snapshot lets hybrid/recurrent models reuse system
// instructions across sessions without attempting an unsafe rollback of conversation state.
static std::vector<llama_token> g_cached_system_prefix_tokens;
static std::vector<uint8_t> g_cached_system_prefix_state;
static llama_model * g_cached_system_prefix_model = nullptr;
static llama_context * g_cached_system_prefix_context = nullptr;
// One validated system-prefix snapshot per local model identity, retained across model switches.
static void clear_android_system_prefix_cache();
static void clear_android_turn_prefix_cache();
static void append_native_trace(const char *text);
struct AndroidSystemPrefixSnapshot { std::vector<llama_token> tokens; std::vector<uint8_t> state; };
static std::unordered_map<std::string, AndroidSystemPrefixSnapshot> g_system_prefix_snapshots;
static std::string g_android_cache_directory;
static constexpr uint32_t kAndroidPrefixCacheSchema = 1;
static constexpr uint32_t kAndroidPrefixCacheMaxTokens = 65536;
static constexpr uint64_t kAndroidPrefixCacheMaxStateBytes = 64ULL * 1024ULL * 1024ULL;

static std::string android_cache_file_path(const std::string & key) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : key) { hash ^= c; hash *= 1099511628211ULL; }
    std::ostringstream name;
    name << std::hex << std::setfill('0') << std::setw(16) << hash << ".kvp";
    return g_android_cache_directory.empty() ? std::string() : g_android_cache_directory + "/" + name.str();
}

static bool read_android_prefix_snapshot_file(const std::string & key, AndroidSystemPrefixSnapshot & out) {
    const std::string path = android_cache_file_path(key);
    if (path.empty()) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char magic[8] = {};
    uint32_t schema = 0, key_size = 0, token_count = 0;
    uint64_t state_size = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&schema), sizeof(schema));
    in.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
    if (!in || std::memcmp(magic, "AICKVP01", 8) != 0 ||
        schema != kAndroidPrefixCacheSchema || key_size == 0 || key_size > 4096) return false;
    std::string stored_key(key_size, '\0');
    in.read(stored_key.data(), static_cast<std::streamsize>(stored_key.size()));
    in.read(reinterpret_cast<char *>(&token_count), sizeof(token_count));
    if (!in || stored_key != key || token_count == 0 || token_count > kAndroidPrefixCacheMaxTokens) return false;
    std::vector<llama_token> tokens(token_count);
    for (uint32_t i = 0; i < token_count; ++i) {
        int32_t token = 0;
        in.read(reinterpret_cast<char *>(&token), sizeof(token));
        if (!in) return false;
        tokens[i] = static_cast<llama_token>(token);
    }
    in.read(reinterpret_cast<char *>(&state_size), sizeof(state_size));
    if (!in || state_size == 0 || state_size > kAndroidPrefixCacheMaxStateBytes) return false;
    std::vector<uint8_t> state(static_cast<size_t>(state_size));
    in.read(reinterpret_cast<char *>(state.data()), static_cast<std::streamsize>(state.size()));
    if (!in || in.peek() != std::char_traits<char>::eof()) return false;
    out = {std::move(tokens), std::move(state)};
    return true;
}

static bool write_android_prefix_snapshot_file(const std::string & key, const AndroidSystemPrefixSnapshot & snapshot) {
    const std::string path = android_cache_file_path(key);
    if (path.empty() || key.empty() || key.size() > 4096 || snapshot.tokens.empty() ||
        snapshot.tokens.size() > kAndroidPrefixCacheMaxTokens || snapshot.state.empty() ||
        snapshot.state.size() > kAndroidPrefixCacheMaxStateBytes) return false;
    const std::string temp = path + ".tmp";
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    const char magic[8] = {'A','I','C','K','V','P','0','1'};
    const uint32_t schema = kAndroidPrefixCacheSchema;
    const uint32_t key_size = static_cast<uint32_t>(key.size());
    const uint32_t token_count = static_cast<uint32_t>(snapshot.tokens.size());
    const uint64_t state_size = static_cast<uint64_t>(snapshot.state.size());
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char *>(&schema), sizeof(schema));
    out.write(reinterpret_cast<const char *>(&key_size), sizeof(key_size));
    out.write(key.data(), static_cast<std::streamsize>(key.size()));
    out.write(reinterpret_cast<const char *>(&token_count), sizeof(token_count));
    for (llama_token token : snapshot.tokens) {
        const int32_t stored_token = static_cast<int32_t>(token);
        out.write(reinterpret_cast<const char *>(&stored_token), sizeof(stored_token));
    }
    out.write(reinterpret_cast<const char *>(&state_size), sizeof(state_size));
    out.write(reinterpret_cast<const char *>(snapshot.state.data()), static_cast<std::streamsize>(snapshot.state.size()));
    out.flush();
    const bool written = out.good();
    out.close();
    if (!written || out.fail()) { std::remove(temp.c_str()); return false; }
    if (std::rename(temp.c_str(), path.c_str()) != 0) { std::remove(temp.c_str()); return false; }
    return true;
}

static void publish_android_system_prefix_snapshot(const std::string & key,
        const std::vector<llama_token> & tokens, const std::vector<uint8_t> & state) {
    if (key.empty() || tokens.empty() || state.empty()) return;
    AndroidSystemPrefixSnapshot snapshot{tokens, state};
    g_system_prefix_snapshots[key] = snapshot;
    const bool saved = write_android_prefix_snapshot_file(key, snapshot);
    append_native_trace((std::string("NATIVE_KV_CACHE_DISK_SAVE ") +
        (saved ? "status=OK" : "status=FAILED") +
        " tokens=" + std::to_string(tokens.size()) +
        " stateBytes=" + std::to_string(state.size())).c_str());
}

static std::string g_active_model_cache_key;
static std::string make_android_model_cache_key(const std::string & path, uint32_t context_length) {
    struct stat st{}; const bool ok = ::stat(path.c_str(), &st) == 0;
    return path + "|size=" + std::to_string(ok ? (long long)st.st_size : -1LL) +
        "|mtime=" + std::to_string(ok ? (long long)st.st_mtime : -1LL) +
        "|ctx=" + std::to_string(context_length) + "|llama=" + AI_CHAT_LLAMA_CPP_SHA;
}
static void preserve_android_system_prefix_cache_for_active_model() {
    if (g_active_model_cache_key.empty() || g_cached_system_prefix_tokens.empty() || g_cached_system_prefix_state.empty()) return;
    publish_android_system_prefix_snapshot(g_active_model_cache_key, g_cached_system_prefix_tokens, g_cached_system_prefix_state);
}
static void set_android_cache_model_key(const std::string & path, uint32_t context_length) {
    g_active_model_cache_key = make_android_model_cache_key(path, context_length);
    clear_android_system_prefix_cache(); clear_android_turn_prefix_cache(); g_cached_prompt_tokens.clear();
    if (g_system_prefix_snapshots.find(g_active_model_cache_key) == g_system_prefix_snapshots.end()) {
        AndroidSystemPrefixSnapshot loaded;
        if (read_android_prefix_snapshot_file(g_active_model_cache_key, loaded)) {
            g_system_prefix_snapshots[g_active_model_cache_key] = std::move(loaded);
            append_native_trace("NATIVE_KV_CACHE_DISK_LOAD status=OK");
        } else {
            append_native_trace("NATIVE_KV_CACHE_DISK_LOAD status=MISS_OR_INVALID");
        }
    }
}
// A second snapshot captures the conversation prefix before the final assistant marker.
// It allows hybrid models to reuse the previous turn's full shared prefix without rollback.
static std::vector<llama_token> g_cached_turn_prefix_tokens;
static std::vector<uint8_t> g_cached_turn_prefix_state;
static llama_model * g_cached_turn_prefix_model = nullptr;
static llama_context * g_cached_turn_prefix_context = nullptr;
// Identity of the model/context that owns the current resident prompt state.
static llama_model * g_cached_model = nullptr;
static llama_context * g_cached_context = nullptr;

static void clear_android_system_prefix_cache() {
    g_cached_system_prefix_tokens.clear();
    g_cached_system_prefix_state.clear();
    g_cached_system_prefix_model = nullptr;
    g_cached_system_prefix_context = nullptr;
}

static void clear_android_turn_prefix_cache() {
    g_cached_turn_prefix_tokens.clear();
    g_cached_turn_prefix_state.clear();
    g_cached_turn_prefix_model = nullptr;
    g_cached_turn_prefix_context = nullptr;
}

static void clear_android_generation_cache() {
    g_cached_prompt_tokens.clear();
    g_cached_model = nullptr;
    g_cached_context = nullptr;
    clear_android_system_prefix_cache();
    clear_android_turn_prefix_cache();
}

#include <signal.h>
#include <unistd.h>
#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>

static std::mutex g_native_runtime_mutex;
static std::atomic<char> g_native_spec_phase[96]{};
static_assert(std::atomic<char>::is_always_lock_free, "Fatal signal speculative phase snapshot requires lock-free atomics");
static void set_spec_phase(const char * phase) {
    const char * value = phase ? phase : "UNKNOWN";
    size_t i = 0;
    while (i + 1 < 96 && value[i]) {
        g_native_spec_phase[i].store(value[i], std::memory_order_relaxed);
        ++i;
    }
    g_native_spec_phase[i].store(0, std::memory_order_relaxed);
}
static void append_native_trace(const char *text);
static void append_weight_residency_trace();

struct NativeDecodeProfile {
    int64_t decode_ms = 0;
    int64_t logits_sync_ms = 0;
    int64_t sampling_ms = 0;
    int64_t callback_ms = 0;
    int64_t token_steps = 0;
    int64_t logits_accesses = 0;
    int64_t callback_calls = 0;
};
static NativeDecodeProfile g_decode_profile;
static void reset_decode_profile() { g_decode_profile = {}; }
static int64_t elapsed_ms(const std::chrono::steady_clock::time_point &a,
                          const std::chrono::steady_clock::time_point &b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}
static void append_decode_profile_trace(int64_t generation_ms, int64_t prefill_ms) {
    const int64_t accounted = g_decode_profile.decode_ms + g_decode_profile.logits_sync_ms +
        g_decode_profile.sampling_ms + g_decode_profile.callback_ms;
    const int64_t decode_window = std::max<int64_t>(0, generation_ms - prefill_ms);
    const int64_t unaccounted = std::max<int64_t>(0, decode_window - accounted);
    append_native_trace((std::string("NATIVE_PERF_PROFILE decodeMs=") +
        std::to_string(g_decode_profile.decode_ms) + " logitsSyncMs=" +
        std::to_string(g_decode_profile.logits_sync_ms) + " samplingMs=" +
        std::to_string(g_decode_profile.sampling_ms) + " callbackMs=" +
        std::to_string(g_decode_profile.callback_ms) + " tokenSteps=" +
        std::to_string(g_decode_profile.token_steps) + " logitsAccesses=" +
        std::to_string(g_decode_profile.logits_accesses) + " callbackCalls=" +
        std::to_string(g_decode_profile.callback_calls) + " accountedMs=" +
        std::to_string(accounted) + " unaccountedMs=" + std::to_string(unaccounted) +
        " decodeWindowMs=" + std::to_string(decode_window)).c_str());
}

// Keep the JNI implementation in native_runtime.cpp. Its own fatal handler is
// retained under a private name; the Android-specific handler below is installed
// explicitly after nativeInit() returns, so it cannot be accidentally shadowed.
#define install_native_fatal_handlers install_native_fatal_handlers_legacy
#include "native_runtime.cpp"
#undef install_native_fatal_handlers

extern "C" JNIEXPORT jboolean JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeSetCacheDirectory(
        JNIEnv *env, jclass, jstring jpath) {
    if (!env || !jpath) return JNI_FALSE;
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) return JNI_FALSE;
    g_android_cache_directory = path;
    env->ReleaseStringUTFChars(jpath, path);
    append_native_trace((std::string("NATIVE_KV_CACHE_DIRECTORY configured=") +
        (g_android_cache_directory.empty() ? "0" : "1")).c_str());
    return g_android_cache_directory.empty() ? JNI_FALSE : JNI_TRUE;
}

static bool model_uses_recurrent_memory() {
    return g_model && (llama_model_is_recurrent(g_model) || llama_model_is_hybrid(g_model));
}

static void android_fatal_signal_handler(int signal_number, siginfo_t * info, void * raw_context) {
    if (g_native_fatal_fd >= 0) {
        const char signal_prefix[] = "NATIVE_FATAL_SIGNAL=";
        const char name_prefix[] = "NATIVE_FATAL_SIGNAL_NAME=";
        const char code_prefix[] = "NATIVE_FATAL_SI_CODE=";
        const char spec_prefix[] = "NATIVE_FATAL_SPEC_PHASE=";
        const char runtime_prefix[] = "NATIVE_FATAL_RUNTIME_PHASE=";
        native_write_text(g_native_fatal_fd, signal_prefix, sizeof(signal_prefix) - 1);

        char value[24];
        int n = 0;
        int v = signal_number;
        if (v == 0) value[n++] = '0';
        else {
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

        const char * signal_name = native_signal_name(signal_number);
        native_write_text(g_native_fatal_fd, name_prefix, sizeof(name_prefix) - 1);
        size_t signal_name_len = 0;
        while (signal_name[signal_name_len]) ++signal_name_len;
        native_write_text(g_native_fatal_fd, signal_name, signal_name_len);
        native_write_text(g_native_fatal_fd, "\n", 1);

        const int signal_code = info ? info->si_code : 0;
        n = 0;
        int code_value = signal_code;
        const bool negative = code_value < 0;
        unsigned int magnitude = negative
            ? (unsigned int)(-(int64_t)code_value)
            : (unsigned int)code_value;
        char reverse_code[24];
        int code_len = 0;
        do {
            reverse_code[code_len++] = (char)('0' + (magnitude % 10u));
            magnitude /= 10u;
        } while (magnitude && code_len < (int)sizeof(reverse_code));
        if (negative) value[n++] = '-';
        while (code_len > 0) value[n++] = reverse_code[--code_len];
        native_write_text(g_native_fatal_fd, code_prefix, sizeof(code_prefix) - 1);
        native_write_text(g_native_fatal_fd, value, (size_t)n);
        native_write_text(g_native_fatal_fd, "\n", 1);

        native_write_text(g_native_fatal_fd, spec_prefix, sizeof(spec_prefix) - 1);
        char spec_value[96];
        size_t spec_len = 0;
        while (spec_len + 1 < sizeof(spec_value)) {
            const char current = g_native_spec_phase[spec_len].load(std::memory_order_relaxed);
            if (!current) break;
            spec_value[spec_len++] = current;
        }
        native_write_text(g_native_fatal_fd, spec_value, spec_len);
        native_write_text(g_native_fatal_fd, "\n", 1);

        native_write_text(g_native_fatal_fd, runtime_prefix, sizeof(runtime_prefix) - 1);
        char runtime_value[128];
        size_t runtime_len = 0;
        while (runtime_len + 1 < sizeof(runtime_value)) {
            const char current = g_native_runtime_phase[runtime_len].load(std::memory_order_relaxed);
            if (!current) break;
            runtime_value[runtime_len++] = current;
        }
        runtime_value[runtime_len] = '\0';
        native_write_text(g_native_fatal_fd, runtime_value, runtime_len);
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
        // si_addr is not meaningful for SIGABRT; avoid recording a misleading pointer.
        if (info && signal_number != SIGABRT) {
            native_write_hex(g_native_fatal_fd, "NATIVE_FATAL_FAULT_ADDR=", (uintptr_t)info->si_addr);
        }
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
    set_spec_phase("IDLE");
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

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeResetStop(
        JNIEnv *, jclass) {
    g_stop.store(false, std::memory_order_release);
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

    const auto callback_started = std::chrono::steady_clock::now();
    env->CallVoidMethod(listener, on_token, chunk);
    const auto callback_finished = std::chrono::steady_clock::now();
    g_decode_profile.callback_ms += elapsed_ms(callback_started, callback_finished);
    ++g_decode_profile.callback_calls;
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
        jint cache_prefix_tokens,
        jint continuation_prefix_tokens,
        jobject listener) {
    std::lock_guard<std::mutex> runtime_lock(g_native_runtime_mutex);
    checkpoint("NATIVE_GENERATE_ENTERED");
    if (!g_context || !g_model) { checkpoint("NATIVE_GENERATE_NO_MODEL"); return 2; }
    if (!jprompt || !listener) { checkpoint("NATIVE_GENERATE_INVALID_ARGUMENT"); return 3; }
    const char * prompt = env->GetStringUTFChars(jprompt, nullptr);
    if (!prompt) { checkpoint("NATIVE_GENERATE_PROMPT_UTF8_FAILED"); return 3; }
    const std::string prompt_text(prompt);
    env->ReleaseStringUTFChars(jprompt, prompt);
    // Stop is reset by the Kotlin flow before its worker is launched. Do not
    // clear it here: cancellation can arrive after launch but before JNI entry.
    if (g_stop.load(std::memory_order_relaxed)) {
        checkpoint("NATIVE_GENERATE_CANCELLED_BEFORE_PREFILL");
        return 9;
    }

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

    // The resident KV sequence is owned by this exact model/context pair.
    // Reloading/replacing either object invalidates all token bookkeeping even
    // when the new model happens to have the same tokenizer.
    if (g_cached_model != g_model || g_cached_context != g_context) {
        if (!g_cached_prompt_tokens.empty()) {
            append_native_trace("NATIVE_KV_CACHE_INVALIDATED reason=model_or_context_replaced");
        }
        g_cached_prompt_tokens.clear();
        clear_android_system_prefix_cache();
        clear_android_turn_prefix_cache();
        g_cached_model = g_model;
        g_cached_context = g_context;
        llama_memory_clear(memory, true);
    }

    size_t common_prefix = 0;
    while (common_prefix < g_cached_prompt_tokens.size() &&
           common_prefix < prompt_tokens.size() &&
           g_cached_prompt_tokens[common_prefix] == prompt_tokens[common_prefix]) {
        ++common_prefix;
    }

    const bool hybrid_memory = model_uses_recurrent_memory();
    const bool cache_supported = !llama_model_has_encoder(g_model);
    const size_t stable_prefix_count = cache_prefix_tokens > 0 &&
            static_cast<size_t>(cache_prefix_tokens) < prompt_tokens.size()
            ? static_cast<size_t>(cache_prefix_tokens) : 0;
    const size_t stable_turn_prefix_count = continuation_prefix_tokens > 0 &&
            static_cast<size_t>(continuation_prefix_tokens) < prompt_tokens.size()
            ? static_cast<size_t>(continuation_prefix_tokens) : 0;
    bool system_prefix_reused = false;
    bool turn_prefix_reused = false;
    bool cache_reused = false;
    size_t reuse_prefix = 0;

    // First try the longest reusable prefix from the previous turn. The saved sequence
    // state ends before the assistant generation marker, so the next turn can append the
    // previous answer and new user turn without unsafe recurrent-memory rollback.
    const bool turn_prefix_matches = hybrid_memory && cache_supported &&
            stable_turn_prefix_count > 0 && !g_cached_turn_prefix_state.empty() &&
            g_cached_turn_prefix_model == g_model &&
            g_cached_turn_prefix_context == g_context &&
            !g_cached_turn_prefix_tokens.empty() &&
            g_cached_turn_prefix_tokens.size() <= stable_turn_prefix_count &&
            std::equal(g_cached_turn_prefix_tokens.begin(),
                       g_cached_turn_prefix_tokens.end(), prompt_tokens.begin());
    if (turn_prefix_matches) {
        llama_memory_clear(memory, true);
        const size_t restored = llama_state_seq_set_data(
                g_context, g_cached_turn_prefix_state.data(),
                g_cached_turn_prefix_state.size(), 0);
        if (restored == g_cached_turn_prefix_state.size()) {
            turn_prefix_reused = true;
            cache_reused = true;
            reuse_prefix = g_cached_turn_prefix_tokens.size();
            common_prefix = reuse_prefix;
            g_cached_prompt_tokens.clear();
            append_native_trace((std::string("NATIVE_KV_CACHE_TURN_PREFIX_RESTORED tokens=") +
                    std::to_string(reuse_prefix) + " stateBytes=" +
                    std::to_string(restored)).c_str());
        } else {
            append_native_trace("NATIVE_KV_CACHE_TURN_PREFIX_RESTORE_FAILED");
            clear_android_turn_prefix_cache();
            llama_memory_clear(memory, true);
        }
    } else if (hybrid_memory && !g_cached_turn_prefix_state.empty()) {
        append_native_trace("NATIVE_KV_CACHE_TURN_PREFIX_INVALIDATED reason=prefix_changed");
        clear_android_turn_prefix_cache();
    }

    // Restore this model's saved system prefix after switching away and back. Model file
    // identity and context are keyed separately, and exact prefix tokens are checked below.
    if (g_cached_system_prefix_state.empty() && !g_active_model_cache_key.empty() && stable_prefix_count > 0) {
        const auto saved = g_system_prefix_snapshots.find(g_active_model_cache_key);
        if (saved != g_system_prefix_snapshots.end() && saved->second.tokens.size() == stable_prefix_count &&
            std::equal(saved->second.tokens.begin(), saved->second.tokens.end(), prompt_tokens.begin())) {
            g_cached_system_prefix_tokens = saved->second.tokens;
            g_cached_system_prefix_state = saved->second.state;
            append_native_trace((std::string("NATIVE_KV_CACHE_MODEL_SNAPSHOT_FOUND tokens=") + std::to_string(stable_prefix_count)).c_str());
        }
    }
    // Session-specific SYSTEM messages remain in the variable suffix and are never replayed.
    if (!turn_prefix_reused && cache_supported && stable_prefix_count > 0 &&
        !g_cached_system_prefix_state.empty() &&
        !g_active_model_cache_key.empty() &&
        g_cached_system_prefix_tokens.size() == stable_prefix_count &&
        std::equal(g_cached_system_prefix_tokens.begin(),
                   g_cached_system_prefix_tokens.end(),
                   prompt_tokens.begin())) {
        llama_memory_clear(memory, true);
        const size_t restored = llama_state_seq_set_data(
                g_context, g_cached_system_prefix_state.data(),
                g_cached_system_prefix_state.size(), 0);
        if (restored == g_cached_system_prefix_state.size()) {
            system_prefix_reused = true;
            cache_reused = true;
            reuse_prefix = stable_prefix_count;
            common_prefix = stable_prefix_count;
            g_cached_prompt_tokens.clear();
            append_native_trace((std::string("NATIVE_KV_CACHE_SYSTEM_PREFIX_RESTORED tokens=") +
                    std::to_string(reuse_prefix) + " stateBytes=" +
                    std::to_string(restored)).c_str());
        } else {
            append_native_trace("NATIVE_KV_CACHE_SYSTEM_PREFIX_RESTORE_FAILED");
            clear_android_system_prefix_cache();
            llama_memory_clear(memory, true);
        }
    } else if (!turn_prefix_reused && !g_cached_system_prefix_state.empty() &&
               (stable_prefix_count == 0 ||
                g_cached_system_prefix_tokens.size() != stable_prefix_count ||
                !std::equal(g_cached_system_prefix_tokens.begin(),
                            g_cached_system_prefix_tokens.end(),
                            prompt_tokens.begin()))) {
        append_native_trace("NATIVE_KV_CACHE_SYSTEM_PREFIX_INVALIDATED reason=prefix_changed");
        clear_android_system_prefix_cache();
    }

    // Sequence-state rollback is not equivalent to ordinary KV trimming for
    // recurrent/hybrid models. Without a system snapshot, their safe fast path
    // remains exact append-only reuse.
    const llama_pos resident_min = llama_memory_seq_pos_min(memory, 0);
    const llama_pos resident_max = llama_memory_seq_pos_max(memory, 0);
    const bool resident_sequence_matches_cache = turn_prefix_reused || system_prefix_reused || (hybrid_memory
            ? (!g_cached_prompt_tokens.empty() &&
               resident_max >= 0 &&
               static_cast<size_t>(resident_max + 1) == g_cached_prompt_tokens.size() &&
               prompt_tokens.size() > g_cached_prompt_tokens.size() &&
               common_prefix == g_cached_prompt_tokens.size())
            : (!g_cached_prompt_tokens.empty() &&
               resident_min == 0 &&
               resident_max >= 0 &&
               static_cast<size_t>(resident_max + 1) == g_cached_prompt_tokens.size()));
    cache_reused = cache_reused ||
            (cache_supported && resident_sequence_matches_cache && common_prefix > 0);
    const bool exact_cached_prefix = !turn_prefix_reused && !system_prefix_reused && cache_reused &&
            common_prefix == g_cached_prompt_tokens.size();
    append_native_trace((std::string("NATIVE_KV_CACHE_PREFIX_CHECK supported=") +
        (cache_supported ? "1" : "0") +
        " hybrid=" + (hybrid_memory ? "1" : "0") +
        " cachedTokens=" + std::to_string((turn_prefix_reused || system_prefix_reused) ? reuse_prefix : g_cached_prompt_tokens.size()) +
        " systemPrefixTokens=" + std::to_string(stable_prefix_count) +
        " continuationPrefixTokens=" + std::to_string(stable_turn_prefix_count) +
        " commonPrefix=" + std::to_string(common_prefix) +
        " residentMin=" + std::to_string(resident_min) +
        " residentMax=" + std::to_string(resident_max) +
        " residentMatches=" + (resident_sequence_matches_cache ? "1" : "0")).c_str());

    if (hybrid_memory) {
        if (turn_prefix_reused || system_prefix_reused) {
            // A validated sequence snapshot was restored; decode only its variable suffix.
        } else if (exact_cached_prefix) {
            // Do not clear or trim recurrent memory. It is still resident from
            // the last successful generation and corresponds to the cached
            // token sequence; process only the new suffix at its true position.
            reuse_prefix = g_cached_prompt_tokens.size();
            append_native_trace((std::string("NATIVE_KV_CACHE_HYBRID_EXACT_REUSE tokens=") +
                std::to_string(reuse_prefix)).c_str());
        } else {
            cache_reused = false;
            reuse_prefix = 0;
            g_cached_prompt_tokens.clear();
            llama_memory_clear(memory, true);
            checkpoint("NATIVE_KV_CACHE_HYBRID_REBUILD");
        }
    } else if (!cache_reused) {
        llama_memory_clear(memory, true);
        g_cached_prompt_tokens.clear();
        checkpoint("NATIVE_KV_CACHE_MISS_CLEARED");
    } else {
        const bool exact_prompt = common_prefix == prompt_tokens.size() &&
                                  common_prefix == g_cached_prompt_tokens.size();
        reuse_prefix = exact_prompt && common_prefix > 0 ? common_prefix - 1 : common_prefix;
        if (!llama_memory_seq_rm(memory, 0, (llama_pos) reuse_prefix, -1)) {
            llama_memory_clear(memory, true);
            g_cached_prompt_tokens.clear();
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

    const size_t diagnostic_cached_tokens = (turn_prefix_reused || system_prefix_reused) ? reuse_prefix : g_cached_prompt_tokens.size();
    const size_t diagnostic_reused_tokens = reuse_prefix;
    const size_t diagnostic_new_tokens = prompt_tokens.size() > reuse_prefix
            ? prompt_tokens.size() - reuse_prefix
            : 0;
    const double diagnostic_hit_ratio = prompt_tokens.empty()
            ? 0.0
            : (100.0 * static_cast<double>(diagnostic_reused_tokens) /
               static_cast<double>(prompt_tokens.size()));
    append_native_trace((std::string("KV Cache: ") +
        (cache_reused ? "HIT" : "MISS")).c_str());
    append_native_trace((std::string("Cached Tokens: ") +
        std::to_string(diagnostic_cached_tokens)).c_str());
    append_native_trace((std::string("Reused Tokens: ") +
        std::to_string(diagnostic_reused_tokens)).c_str());
    append_native_trace((std::string("New Tokens: ") +
        std::to_string(diagnostic_new_tokens)).c_str());
    append_native_trace((std::string("Cache Hit Ratio: ") +
        std::to_string(diagnostic_hit_ratio) + "%").c_str());
    append_native_trace((std::string("NATIVE_KV_CACHE_REQUEST status=") +
        (cache_reused ? "HIT" : "MISS") +
        " cachedTokens=" + std::to_string(diagnostic_cached_tokens) +
        " systemPrefixTokens=" + std::to_string(stable_prefix_count) +
        " continuationPrefixTokens=" + std::to_string(stable_turn_prefix_count) +
        " cacheMode=" + (turn_prefix_reused ? "turn_prefix_snapshot" : (system_prefix_reused ? "system_prefix_snapshot" : (hybrid_memory ? "resident_hybrid" : "resident_kv"))) +
        " reusedTokens=" + std::to_string(diagnostic_reused_tokens) +
        " newTokens=" + std::to_string(diagnostic_new_tokens) +
        " hitRatio=" + std::to_string(diagnostic_hit_ratio)).c_str());

    // The context is intentionally configured with a bounded physical batch/ubatch on Android.
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
        clear_android_generation_cache();
        llama_memory_clear(memory, true);
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
        std::to_string(decode_batch_size) + " nUbatch=" +
        std::to_string(llama_n_ubatch(g_context)) + " promptTokens=" +
        std::to_string(n_prompt) + " reusedTokens=" +
        std::to_string(reuse_prefix)).c_str());

    int generated = 0;
    // Only tokens that have actually been decoded into g_context belong in the
    // resident cache. The sampled token that reaches EOS/max-tokens before the
    // next llama_decode() is intentionally not included.
    std::vector<llama_token> generated_decoded_tokens;
    generated_decoded_tokens.reserve((size_t) max_predict);
    int position = (int) reuse_prefix;
    std::string pending_utf8;
    const char * stop_reason = "UNKNOWN";
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
        if (g_stop.load(std::memory_order_relaxed)) { result = 9; stop_reason = "USER_STOP"; checkpoint("NATIVE_GENERATE_STOPPED"); break; }
        int count = std::min(decode_batch_size, n_prompt - prompt_offset);
        const bool capture_system_prefix = cache_supported && stable_prefix_count > reuse_prefix &&
                g_cached_system_prefix_state.empty();
        const bool capture_turn_prefix = hybrid_memory &&
                stable_turn_prefix_count > reuse_prefix;
        // Split at the next reusable checkpoint boundary. Save system state first for
        // new-session reuse, then save the longer per-turn prefix before assistant decode.
        size_t next_snapshot_boundary = static_cast<size_t>(n_prompt);
        if (capture_system_prefix && stable_prefix_count > static_cast<size_t>(prompt_offset)) {
            next_snapshot_boundary = std::min(next_snapshot_boundary, stable_prefix_count);
        }
        if (capture_turn_prefix && stable_turn_prefix_count > static_cast<size_t>(prompt_offset)) {
            next_snapshot_boundary = std::min(next_snapshot_boundary, stable_turn_prefix_count);
        }
        if (next_snapshot_boundary < static_cast<size_t>(prompt_offset + count)) {
            count = static_cast<int>(next_snapshot_boundary - static_cast<size_t>(prompt_offset));
        }
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
        if (capture_system_prefix && prompt_offset == static_cast<int>(stable_prefix_count)) {
            const size_t state_size = llama_state_seq_get_size(g_context, 0);
            if (state_size > 0) {
                std::vector<uint8_t> state(state_size);
                const size_t copied = llama_state_seq_get_data(
                        g_context, state.data(), state.size(), 0);
                if (copied > 0 && copied <= state.size()) {
                    state.resize(copied);
                    g_cached_system_prefix_state = std::move(state);
                    g_cached_system_prefix_tokens.assign(
                            prompt_tokens.begin(),
                            prompt_tokens.begin() + static_cast<std::ptrdiff_t>(stable_prefix_count));
                    g_cached_system_prefix_model = g_model;
                    g_cached_system_prefix_context = g_context;
                    if (!g_active_model_cache_key.empty()) publish_android_system_prefix_snapshot(g_active_model_cache_key, g_cached_system_prefix_tokens, g_cached_system_prefix_state);
                    append_native_trace((std::string("NATIVE_KV_CACHE_SYSTEM_PREFIX_SAVED tokens=") +
                            std::to_string(stable_prefix_count) + " stateBytes=" +
                            std::to_string(copied)).c_str());
                } else {
                    append_native_trace("NATIVE_KV_CACHE_SYSTEM_PREFIX_SAVE_FAILED");
                    clear_android_system_prefix_cache();
                }
            } else {
                append_native_trace("NATIVE_KV_CACHE_SYSTEM_PREFIX_SAVE_SKIPPED reason=empty_state");
            }
        }
        if (capture_turn_prefix && prompt_offset == static_cast<int>(stable_turn_prefix_count)) {
            const size_t state_size = llama_state_seq_get_size(g_context, 0);
            if (state_size > 0) {
                std::vector<uint8_t> state(state_size);
                const size_t copied = llama_state_seq_get_data(
                        g_context, state.data(), state.size(), 0);
                if (copied > 0 && copied <= state.size()) {
                    state.resize(copied);
                    g_cached_turn_prefix_state = std::move(state);
                    g_cached_turn_prefix_tokens.assign(
                            prompt_tokens.begin(),
                            prompt_tokens.begin() + static_cast<std::ptrdiff_t>(stable_turn_prefix_count));
                    g_cached_turn_prefix_model = g_model;
                    g_cached_turn_prefix_context = g_context;
                    append_native_trace((std::string("NATIVE_KV_CACHE_TURN_PREFIX_SAVED tokens=") +
                            std::to_string(stable_turn_prefix_count) + " stateBytes=" +
                            std::to_string(copied)).c_str());
                } else {
                    append_native_trace("NATIVE_KV_CACHE_TURN_PREFIX_SAVE_FAILED");
                    clear_android_turn_prefix_cache();
                }
            } else {
                append_native_trace("NATIVE_KV_CACHE_TURN_PREFIX_SAVE_SKIPPED reason=empty_state");
            }
        }
        append_native_trace((std::string("NATIVE_PREFILL_PROGRESS processed=") +
                std::to_string(prompt_offset) + " total=" + std::to_string(n_prompt) +
                " reused=" + std::to_string(reuse_prefix)).c_str());
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

    // Keep the live recurrent/KV state resident in g_context. We intentionally
    // avoid sequence-state serialization here; the next turn can reuse this
    // resident state directly when its token prefix matches the cached tokens.

    reset_decode_profile();
    if (result == 0) {
        while (generated < max_predict) {
            if (g_stop.load(std::memory_order_relaxed)) { result = 9; stop_reason = "USER_STOP"; checkpoint("NATIVE_GENERATE_STOPPED"); break; }

            append_native_trace((std::string("NATIVE_DECODE_STEP_STARTED step=") + std::to_string(generated + 1)).c_str());
            const auto logits_sync_started = std::chrono::steady_clock::now();
            append_native_trace("NATIVE_LOGITS_SYNC_START");
            const float * logits_probe = llama_get_logits_ith(g_context, -1);
            const auto logits_sync_finished = std::chrono::steady_clock::now();
            g_decode_profile.logits_sync_ms += elapsed_ms(logits_sync_started, logits_sync_finished);
            ++g_decode_profile.logits_accesses;
            append_native_trace((std::string("NATIVE_LOGITS_SYNC_END elapsedMs=") + std::to_string(elapsed_ms(logits_sync_started, logits_sync_finished))).c_str());
            if (!logits_probe) append_native_trace("NATIVE_LOGITS_ACCESS_NULL");

            append_native_trace("NATIVE_SAMPLING_START");
            const auto sampling_started = std::chrono::steady_clock::now();
            const llama_token token = llama_sampler_sample(sampler, g_context, -1);
            llama_sampler_accept(sampler, token);
            const auto sampling_finished = std::chrono::steady_clock::now();
            g_decode_profile.sampling_ms += elapsed_ms(sampling_started, sampling_finished);
            ++g_decode_profile.token_steps;
            append_native_trace((std::string("NATIVE_SAMPLING_END elapsedMs=") + std::to_string(elapsed_ms(sampling_started, sampling_finished))).c_str());
            append_native_trace((std::string("NATIVE_TOKEN_SELECTED token=") + std::to_string((int)token)).c_str());
            if (llama_vocab_is_eog(vocab, token)) { stop_reason = "EOS"; checkpoint("NATIVE_GENERATE_EOG"); break; }

            char piece[1024];
            int piece_size = llama_token_to_piece(vocab, token, piece, (int) sizeof(piece), 0, true);
            if (piece_size < 0) {
                append_native_trace((std::string("NATIVE_GENERATE_TOKEN_TO_PIECE_FAILED size=") + std::to_string(piece_size)).c_str());
                checkpoint("NATIVE_GENERATE_TOKEN_TO_PIECE_FAILED");
                result = 11;
                stop_reason = "TOKEN_TO_PIECE_ERROR";
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
                    stop_reason = "STREAM_CALLBACK_ERROR";
                    break;
                }
                tokens_since_emit = 0;
            }
            if (generated >= max_predict) { stop_reason = requested_max > max_predict ? "CONTEXT_LIMIT" : "MAX_TOKENS"; break; }

            // Decode exactly one sampled token. This keeps generation memory bounded
            // while retaining the full prompt/context and without changing sampling.
            fill_batch(&token, 1, position, true);
            const auto decode_started = std::chrono::steady_clock::now();
            const int decode_result = llama_decode(g_context, batch);
            const auto decode_finished = std::chrono::steady_clock::now();
            g_decode_profile.decode_ms += elapsed_ms(decode_started, decode_finished);
            append_native_trace((std::string("NATIVE_DECODE_STEP_END elapsedMs=") + std::to_string(elapsed_ms(decode_started, decode_finished)) +
                " result=" + std::to_string(decode_result)).c_str());
            if (decode_result != 0) {
                append_native_trace((std::string("NATIVE_GENERATE_DECODE_FAILED code=") + std::to_string(decode_result) +
                    " generated=" + std::to_string(generated)).c_str());
                g_cached_prompt_tokens.clear();
                llama_memory_clear(memory, true);
                checkpoint("NATIVE_GENERATE_DECODE_FAILED_CACHE_RESET");
                result = 10;
                stop_reason = "DECODE_ERROR";
                break;
            }
            generated_decoded_tokens.push_back(token);
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
    const auto profile_prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_finished_at - generation_started_at).count();
    append_decode_profile_trace(generation_ms, profile_prefill_ms);
    append_native_trace((std::string("NATIVE_EXECUTION_PROFILE mode=") +
        (g_gpu && g_webgpu_backend_loaded ? "WEBGPU_GPU_PREFERRED_MIXED_CPU_OPS" : "CPU_FALLBACK") +
        " gpuBackendRegistered=" + (g_webgpu_backend_loaded ? "1" : "0") +
        " selectedBackend=" + (g_gpu && g_webgpu_backend_loaded ? "WEBGPU" : "CPU") +
        " fallbackPolicy=ggml_scheduler cpuFallbackAllowed=1" +
        " decodeSteps=" + std::to_string(g_decode_profile.token_steps) +
        " decodeWallMs=" + std::to_string(g_decode_profile.decode_ms) +
        " cpuLogitsSyncMs=" + std::to_string(g_decode_profile.logits_sync_ms) +
        " cpuSamplingMs=" + std::to_string(g_decode_profile.sampling_ms) +
        " cpuCallbackMs=" + std::to_string(g_decode_profile.callback_ms) +
        " cpuMeasuredSubtotalMs=" + std::to_string(g_decode_profile.logits_sync_ms +
            g_decode_profile.sampling_ms + g_decode_profile.callback_ms) +
        " gpuKernelTiming=not_instrumented").c_str());
    if (prefill_recorded && result == 0) {
        // Publish only the tokens that are actually resident in the live
        // context. For hybrid models, retain the full recurrent+attention state
        // in place and allow exact-prefix append-only reuse on the next turn.
        g_cached_prompt_tokens = prompt_tokens;
        g_cached_prompt_tokens.insert(
                g_cached_prompt_tokens.end(),
                generated_decoded_tokens.begin(),
                generated_decoded_tokens.end());
        g_cached_model = g_model;
        g_cached_context = g_context;
        llama_pos published_min = llama_memory_seq_pos_min(memory, 0);
        llama_pos published_max = llama_memory_seq_pos_max(memory, 0);
        // Hybrid/recurrent memory cannot be safely trimmed or rolled back. If its
        // resident position drifted from the token ledger, rebuild the resident
        // state from the exact token sequence rather than dropping cache reuse
        // for every following turn.
        if (hybrid_memory && !g_cached_prompt_tokens.empty() &&
            (published_max < 0 ||
             static_cast<size_t>(published_max + 1) != g_cached_prompt_tokens.size())) {
            append_native_trace("NATIVE_KV_CACHE_HYBRID_RESYNC_STARTED");
            llama_memory_clear(memory, true);
            llama_batch replay_batch = llama_batch_init(decode_batch_size, 0, 1);
            bool replay_ok = replay_batch.token && replay_batch.pos &&
                    replay_batch.n_seq_id && replay_batch.seq_id && replay_batch.logits;
            size_t replay_offset = 0;
            while (replay_ok && replay_offset < g_cached_prompt_tokens.size()) {
                const int count = static_cast<int>(std::min(
                        static_cast<size_t>(decode_batch_size),
                        g_cached_prompt_tokens.size() - replay_offset));
                replay_batch.n_tokens = count;
                for (int i = 0; i < count; ++i) {
                    replay_batch.token[i] = g_cached_prompt_tokens[replay_offset + static_cast<size_t>(i)];
                    replay_batch.pos[i] = static_cast<llama_pos>(replay_offset + static_cast<size_t>(i));
                    replay_batch.n_seq_id[i] = 1;
                    replay_batch.seq_id[i][0] = 0;
                    replay_batch.logits[i] = 0;
                }
                if (llama_decode(g_context, replay_batch) != 0) {
                    replay_ok = false;
                    break;
                }
                replay_offset += static_cast<size_t>(count);
            }
            llama_batch_free(replay_batch);
            published_min = llama_memory_seq_pos_min(memory, 0);
            published_max = llama_memory_seq_pos_max(memory, 0);
            replay_ok = replay_ok && published_max >= 0 &&
                    static_cast<size_t>(published_max + 1) == g_cached_prompt_tokens.size();
            append_native_trace((std::string("NATIVE_KV_CACHE_HYBRID_RESYNC_RESULT success=") +
                    (replay_ok ? "1" : "0") +
                    " replayedTokens=" + std::to_string(replay_offset) +
                    " residentMax=" + std::to_string(published_max) +
                    " cachedTokens=" + std::to_string(g_cached_prompt_tokens.size())).c_str());
            if (!replay_ok) {
                clear_android_generation_cache();
                llama_memory_clear(memory, true);
            }
        }
        const bool resident_publish_valid = hybrid_memory
                ? (!g_cached_prompt_tokens.empty() &&
                   published_max >= 0 &&
                   static_cast<size_t>(published_max + 1) == g_cached_prompt_tokens.size())
                : (published_min == 0 &&
                   published_max >= 0 &&
                   static_cast<size_t>(published_max + 1) == g_cached_prompt_tokens.size());
        if (!resident_publish_valid) {
            append_native_trace((std::string("NATIVE_KV_CACHE_PUBLISH_INVALID residentMin=") +
                std::to_string(published_min) + " residentMax=" +
                std::to_string(published_max) + " cachedTokens=" +
                std::to_string(g_cached_prompt_tokens.size()) +
                " hybrid=" + (hybrid_memory ? "1" : "0")).c_str());
            clear_android_generation_cache();
            llama_memory_clear(memory, true);
        } else {
            append_native_trace((std::string("NATIVE_KV_CACHE_PUBLISHED tokens=") +
                std::to_string(g_cached_prompt_tokens.size()) +
                " promptTokens=" + std::to_string(prompt_tokens.size()) +
                " generatedDecodedTokens=" + std::to_string(generated_decoded_tokens.size()) +
                " residentMin=" + std::to_string(published_min) +
                " residentMax=" + std::to_string(published_max) +
                " mode=" + (hybrid_memory ? (!g_cached_turn_prefix_state.empty() ? "resident_hybrid_turn_prefix_snapshot" : (!g_cached_system_prefix_state.empty() ? "resident_hybrid_system_prefix_snapshot" : "resident_hybrid_exact_prefix")) : "resident_kv") +
                " systemPrefixTokens=" + std::to_string(g_cached_system_prefix_tokens.size()) +
                " continuationPrefixTokens=" + std::to_string(g_cached_turn_prefix_tokens.size())).c_str());
        }
    } else if (result != 0) {
        // Any interrupted/failed request may have trimmed or partially changed
        // the resident memory, even when prefill never reached its completion
        // marker. Invalidate both sides together to prevent a stale token/state pair.
        clear_android_generation_cache();
        llama_memory_clear(memory, true);
        checkpoint("NATIVE_KV_CACHE_NOT_PUBLISHED_AFTER_FAILED_GENERATION");
    }
    if (stop_reason == std::string("UNKNOWN") && result == 0) {
        stop_reason = generated >= max_predict
                ? (requested_max > max_predict ? "CONTEXT_LIMIT" : "MAX_TOKENS")
                : "COMPLETED";
    }
    append_weight_residency_trace();
    append_native_trace((std::string("NATIVE_KV_CACHE_RESULT status=") +
        (cache_reused ? "HIT" : "MISS") +
        " cachedTokens=" + std::to_string(diagnostic_cached_tokens) +
        " reusedTokens=" + std::to_string(diagnostic_reused_tokens) +
        " newTokens=" + std::to_string(diagnostic_new_tokens) +
        " hitRatio=" + std::to_string(diagnostic_hit_ratio)).c_str());

    append_native_trace((std::string("NATIVE_STOP_REASON reason=") + stop_reason +
        " generatedTokens=" + std::to_string(generated) +
        " requestedMaxTokens=" + std::to_string(requested_max) +
        " effectiveMaxTokens=" + std::to_string(max_predict)).c_str());
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
