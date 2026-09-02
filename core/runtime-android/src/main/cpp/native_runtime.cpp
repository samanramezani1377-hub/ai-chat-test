#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "llama.h"

#define LOG_TAG "AIChatRuntime"
static std::atomic_bool g_stop{false};
static llama_model *g_model = nullptr;
static llama_context *g_context = nullptr;
static llama_sampler *g_sampler = nullptr;
static bool g_gpu = false;

static bool abort_callback(void *) { return g_stop.load(std::memory_order_relaxed); }
static int threads() { return std::clamp((int)std::max(1u, std::thread::hardware_concurrency()) - 2, 2, 4); }
static void free_all() {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_context) { llama_free(g_context); g_context = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_gpu = false;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeInit(JNIEnv *, jclass) {
    llama_log_set([](enum ggml_log_level level, const char *text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "%s", text);
    }, nullptr);
    ggml_backend_load_all();
    llama_backend_init();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeLoad(JNIEnv *env, jclass, jstring jpath, jint ctx_len, jint gpu_layers) {
    free_all();
    g_stop.store(false);
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    g_model = llama_model_load_from_file(path, mp);
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model && gpu_layers != 0) {
        path = env->GetStringUTFChars(jpath, nullptr);
        mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        g_model = llama_model_load_from_file(path, mp);
        env->ReleaseStringUTFChars(jpath, path);
        g_gpu = false;
    } else g_gpu = g_model != nullptr && gpu_layers != 0;
    if (!g_model) return 1;

    llama_context_params cp = llama_context_default_params();
    const int trained = llama_model_n_ctx_train(g_model);
    cp.n_ctx = (uint32_t)std::max(1, std::min(ctx_len > 0 ? ctx_len : 4096, trained));
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, 512);
    cp.n_ubatch = cp.n_batch;
    cp.n_threads = threads();
    cp.n_threads_batch = threads();
    g_context = llama_init_from_model(g_model, cp);
    if (!g_context) { free_all(); return 2; }
    llama_set_abort_callback(g_context, abort_callback, nullptr);
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
