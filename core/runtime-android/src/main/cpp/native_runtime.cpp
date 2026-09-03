#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "llama.h"

#define LOG_TAG "AIChatRuntime"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static std::atomic_bool g_stop{false};
static llama_model *g_model = nullptr;
static llama_context *g_context = nullptr;
static llama_sampler *g_sampler = nullptr;
static bool g_gpu = false;
static std::string g_native_marker_file;
static std::mutex g_native_marker_mutex;

static bool abort_callback(void *) { return g_stop.load(std::memory_order_relaxed); }
static int threads() { return std::clamp((int)std::max(1u, std::thread::hardware_concurrency()) - 2, 2, 4); }

static void checkpoint(const char *event) {
    LOGI("NATIVE_CHECKPOINT %s", event);
    __android_log_write(ANDROID_LOG_INFO, LOG_TAG, event);
    if (!g_native_marker_file.empty()) {
        std::lock_guard<std::mutex> lock(g_native_marker_mutex);
        std::ofstream out(g_native_marker_file, std::ios::trunc);
        if (out.is_open()) {
            out << event << '\n';
            out.flush();
        }
    }
}

static void free_all() {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_context) { llama_free(g_context); g_context = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_gpu = false;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeInit(JNIEnv *env, jclass) {
    LOGI("ACTIVATION_NATIVE_INIT_STARTED");

    jclass system_class = env->FindClass("java/lang/System");
    if (system_class) {
        jmethodID get_property = env->GetStaticMethodID(system_class, "getProperty", "(Ljava/lang/String;)Ljava/lang/String;");
        if (get_property) {
            jstring key = env->NewStringUTF("java.io.tmpdir");
            jstring tmp_dir = static_cast<jstring>(env->CallStaticObjectMethod(system_class, get_property, key));
            env->DeleteLocalRef(key);
            if (tmp_dir) {
                const char *dir = env->GetStringUTFChars(tmp_dir, nullptr);
                if (dir) {
                    g_native_marker_file = std::string(dir) + "/ai-chat-last-native-event.txt";
                    env->ReleaseStringUTFChars(tmp_dir, dir);
                }
                env->DeleteLocalRef(tmp_dir);
            }
        }
        env->DeleteLocalRef(system_class);
    }

    if (!g_native_marker_file.empty()) {
        LOGI("NATIVE_DIAGNOSTICS_MARKER=%s", g_native_marker_file.c_str());
    } else {
        LOGW("NATIVE_DIAGNOSTICS_MARKER_UNAVAILABLE");
    }

    llama_log_set([](enum ggml_log_level level, const char *text, void *) {
        const int priority = level >= GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR :
                             level >= GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
        __android_log_print(priority, LOG_TAG, "%s", text ? text : "<null>");
    }, nullptr);
    LOGI("ACTIVATION_BACKEND_LOAD_ALL_STARTED");
    ggml_backend_load_all();
    LOGI("ACTIVATION_BACKEND_LOAD_ALL_RETURNED");
    LOGI("ACTIVATION_LLAMA_BACKEND_INIT_STARTED");
    llama_backend_init();
    LOGI("ACTIVATION_LLAMA_BACKEND_INIT_RETURNED");
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeLoad(JNIEnv *env, jclass, jstring jpath, jint ctx_len, jint gpu_layers) {
    checkpoint("NATIVE_LOAD_STARTED");
    LOGI("ACTIVATION_NATIVE_LOAD_STARTED ctx_len=%d gpu_layers=%d", (int)ctx_len, (int)gpu_layers);
    LOGI("ACTIVATION_FREE_OLD_RUNTIME_STARTED");
    checkpoint("FREE_OLD_RUNTIME_STARTED");
    free_all();
    LOGI("ACTIVATION_FREE_OLD_RUNTIME_RETURNED");
    checkpoint("FREE_OLD_RUNTIME_RETURNED");
    g_stop.store(false);

    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) { LOGE("ACTIVATION_PATH_UTF8_FAILED"); checkpoint("PATH_UTF8_FAILED"); return 3; }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    LOGI("ACTIVATION_MODEL_LOAD_STARTED gpu_layers=%d", (int)mp.n_gpu_layers);
    checkpoint(gpu_layers == 0 ? "MODEL_LOAD_STARTED gpu_layers=0" : "MODEL_LOAD_STARTED gpu_layers=GPU");
    g_model = llama_model_load_from_file(path, mp);
    LOGI("ACTIVATION_MODEL_LOAD_RETURNED success=%d", g_model != nullptr ? 1 : 0);
    checkpoint(g_model ? "MODEL_LOAD_RETURNED_SUCCESS" : "MODEL_LOAD_RETURNED_FAILED");
    env->ReleaseStringUTFChars(jpath, path);

    if (!g_model && gpu_layers != 0) {
        LOGW("ACTIVATION_GPU_MODEL_LOAD_FAILED_STARTING_CPU_FALLBACK");
        checkpoint("GPU_MODEL_LOAD_FAILED_CPU_FALLBACK_STARTED");
        path = env->GetStringUTFChars(jpath, nullptr);
        if (!path) { LOGE("ACTIVATION_CPU_FALLBACK_PATH_UTF8_FAILED"); checkpoint("CPU_FALLBACK_PATH_UTF8_FAILED"); return 3; }
        mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        LOGI("ACTIVATION_CPU_MODEL_LOAD_STARTED");
        checkpoint("CPU_MODEL_LOAD_STARTED");
        g_model = llama_model_load_from_file(path, mp);
        LOGI("ACTIVATION_CPU_MODEL_LOAD_RETURNED success=%d", g_model != nullptr ? 1 : 0);
        checkpoint(g_model ? "CPU_MODEL_LOAD_RETURNED_SUCCESS" : "CPU_MODEL_LOAD_RETURNED_FAILED");
        env->ReleaseStringUTFChars(jpath, path);
        g_gpu = false;
    } else {
        g_gpu = g_model != nullptr && gpu_layers != 0;
    }

    if (!g_model) { LOGE("ACTIVATION_MODEL_LOAD_FAILED"); checkpoint("MODEL_LOAD_FAILED"); return 1; }

    const int trained = llama_model_n_ctx_train(g_model);
    const int requested = ctx_len > 0 ? ctx_len : 4096;
    const int effective = std::max(1, std::min(requested, trained));
    LOGI("ACTIVATION_MODEL_READY trained_ctx=%d requested_ctx=%d effective_ctx=%d backend=%s", trained, requested, effective, g_gpu ? "Vulkan" : "CPU");
    checkpoint("MODEL_READY");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)effective;
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, 512);
    cp.n_ubatch = cp.n_batch;
    cp.n_threads = threads();
    cp.n_threads_batch = threads();
    LOGI("ACTIVATION_CONTEXT_INIT_STARTED ctx=%u batch=%u ubatch=%u threads=%d", cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_threads);
    checkpoint("CONTEXT_INIT_STARTED");
    g_context = llama_init_from_model(g_model, cp);
    LOGI("ACTIVATION_CONTEXT_INIT_RETURNED success=%d", g_context != nullptr ? 1 : 0);
    checkpoint(g_context ? "CONTEXT_INIT_RETURNED_SUCCESS" : "CONTEXT_INIT_RETURNED_FAILED");
    if (!g_context) { LOGE("ACTIVATION_CONTEXT_INIT_FAILED"); checkpoint("CONTEXT_INIT_FAILED"); free_all(); return 2; }
    llama_set_abort_callback(g_context, abort_callback, nullptr);
    LOGI("ACTIVATION_NATIVE_LOAD_COMPLETED backend=%s ctx=%d", g_gpu ? "Vulkan" : "CPU", (int)llama_n_ctx(g_context));
    checkpoint("NATIVE_LOAD_COMPLETED");
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeGenerate(JNIEnv *env, jclass, jstring jprompt,
        jint max_tokens, jfloat temperature, jint top_k, jfloat top_p, jfloat min_p, jobject listener) {
    if (!g_model || !g_context) return 1;
    g_stop.store(false);
    llama_memory_clear(llama_get_memory(g_context), true);

    const char *prompt = env->GetStringUTFChars(jprompt, nullptr);
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    const int n = -llama_tokenize(vocab, prompt, strlen(prompt), nullptr, 0, true, true);
    if (n <= 0) { env->ReleaseStringUTFChars(jprompt, prompt); return 2; }
    std::vector<llama_token> tokens(n);
    if (llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true) < 0) {
        env->ReleaseStringUTFChars(jprompt, prompt); return 3;
    }
    env->ReleaseStringUTFChars(jprompt, prompt);

    if (g_sampler) llama_sampler_free(g_sampler);
    g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (top_k > 0) llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_min_p(min_p, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(std::max(0.0f, temperature)));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    jmethodID on_token = nullptr;
    if (listener) on_token = env->GetMethodID(env->GetObjectClass(listener), "onToken", "(Ljava/lang/String;)V");

    const int batch_size = (int)llama_n_batch(g_context);
    for (int offset = 0; offset < n; offset += batch_size) {
        const int count = std::min(batch_size, n - offset);
        llama_batch batch = llama_batch_get_one(tokens.data() + offset, count);
        if (llama_memory_seq_pos_max(llama_get_memory(g_context), 0) + 1 + batch.n_tokens > (int)llama_n_ctx(g_context)) return 4;
        const int rc = llama_decode(g_context, batch);
        if (rc != 0) return rc == 2 ? 5 : 6;
    }

    int generated = 0;
    while (generated < std::max(1, max_tokens) && !g_stop.load()) {
        const llama_token id = llama_sampler_sample(g_sampler, g_context, -1);
        llama_sampler_accept(g_sampler, id);
        if (llama_vocab_is_eog(vocab, id)) break;
        char piece[512];
        const int bytes = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, true);
        if (bytes < 0) return 7;
        if (listener && on_token && bytes > 0) {
            std::string text(piece, bytes);
            env->CallVoidMethod(listener, on_token, env->NewStringUTF(text.c_str()));
            if (env->ExceptionCheck()) { env->ExceptionClear(); return 8; }
        }
        ++generated;
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&id), 1);
        if (llama_memory_seq_pos_max(llama_get_memory(g_context), 0) + 1 + batch.n_tokens > (int)llama_n_ctx(g_context)) return 4;
        const int rc = llama_decode(g_context, batch);
        if (rc != 0) return rc == 2 ? 5 : 6;
    }
    return g_stop.load() ? 9 : 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeStop(JNIEnv *, jclass) { g_stop.store(true); }

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeUnload(JNIEnv *, jclass) { g_stop.store(true); free_all(); }

extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeRuntimeInfo(JNIEnv *env, jclass) {
    const std::string value = std::string(g_gpu ? "Hybrid(CPU+Vulkan)" : "CPU/NEON") + "; llama.cpp=" + AI_CHAT_LLAMA_CPP_SHA;
    return env->NewStringUTF(value.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeContextLength(JNIEnv *, jclass) {
    return g_context ? (jint)llama_n_ctx(g_context) : 0;
}
