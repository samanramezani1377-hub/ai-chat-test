#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include <vector>

// Tokens represented by the resident prompt prefix in the native KV cache.
static std::vector<llama_token> g_cached_prompt_tokens;
// Identity of the model/context that owns the resident KV cache.
static llama_model * g_cached_model = nullptr;
static llama_context * g_cached_context = nullptr;
// Hybrid/recurrent models cannot safely trim their recurrent state with
// llama_memory_seq_rm(). Keep an exact post-prefill sequence snapshot instead.
static std::vector<uint8_t> g_cached_prompt_state;
// ON_DEVICE sequence states keep large recurrent tensors in backend device buffers.
// The host vector then contains only the small serialized state metadata.
static bool g_cached_prompt_state_on_device = false;
static void clear_android_generation_cache() {
    g_cached_prompt_tokens.clear();
    g_cached_prompt_state.clear();
    g_cached_prompt_state_on_device = false;
    g_cached_model = nullptr;
    g_cached_context = nullptr;
}

#include <signal.h>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>
#include <map>
#include <thread>
#include <algorithm>
#include <cmath>
#include <sched.h>
#include <cstdio>

static std::mutex g_native_runtime_mutex;
static bool g_spec_requested=false;
static bool g_spec_mtp=false;
static std::string g_spec_draft_path;
static std::string g_target_model_path;
static double g_spec_accept_ema=1.0;
static common_speculative_init_result_ptr g_spec_init;
static common_speculative * g_spec = nullptr;
static volatile const char * g_spec_phase = "IDLE";
static void set_spec_phase(const char * phase) { g_spec_phase = phase ? phase : "UNKNOWN"; }
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

static void append_speculative_stats_trace(int draft_tokens, int accepted_tokens, int steps) {
    const double rate = draft_tokens > 0 ? (100.0 * static_cast<double>(accepted_tokens) / static_cast<double>(draft_tokens)) : 0.0;
    const double mean = steps > 0 ? (static_cast<double>(accepted_tokens) / static_cast<double>(steps)) : 0.0;
    append_native_trace((std::string("SPECULATIVE_STATS draftTokens=") + std::to_string(draft_tokens) +
        " acceptedTokens=" + std::to_string(accepted_tokens) +
        " acceptanceRate=" + std::to_string(rate) +
        " steps=" + std::to_string(steps) +
        " meanAcceptedPerStep=" + std::to_string(mean)).c_str());
}

// Keep the JNI implementation in native_runtime.cpp. Its own fatal handler is
// retained under a private name; the Android-specific handler below is installed
// explicitly after nativeInit() returns, so it cannot be accidentally shadowed.
#define install_native_fatal_handlers install_native_fatal_handlers_legacy
#include "native_runtime.cpp"
#undef install_native_fatal_handlers

// Adapt CPU workers from measured inference on this device. OpenCL-selected and
// speculative generations keep their existing threading policy.
struct CpuThreadTuningState { int attempts = 0; int samples = 0; double ema_tokens_per_sec = 0.0; };
static std::map<int, CpuThreadTuningState> g_cpu_thread_tuning;
static int g_cpu_thread_trial_count = 0;
static int g_cpu_default_threads = 0;
static llama_context * g_cpu_tuned_context = nullptr;
static std::string g_cpu_tuning_key;
static std::string g_cpu_tuning_path;
static unsigned cpu_threads_available_to_process();

// Use a deterministic key so learned settings survive process restarts but never
// leak from one model/context/device configuration to another.
static uint64_t cpu_tuning_hash(const std::string &value) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char ch : value) { hash ^= ch; hash *= 1099511628211ULL; }
    return hash;
}
static std::string cpu_tuning_identity(llama_context *ctx) {
    const std::string model = g_target_model_path.empty() ? "<unknown-model>" : g_target_model_path;
    struct stat model_stat{};
    const bool model_stat_ok = !g_target_model_path.empty() &&
        stat(g_target_model_path.c_str(), &model_stat) == 0;
    const uint64_t hwcap = static_cast<uint64_t>(getauxval(AT_HWCAP));
    const uint64_t hwcap2 = static_cast<uint64_t>(getauxval(AT_HWCAP2));
    const unsigned allowed_cores = cpu_threads_available_to_process();
    return model + "|size=" + std::to_string(model_stat_ok ? model_stat.st_size : -1) +
        "|mtime=" + std::to_string(model_stat_ok ? model_stat.st_mtime : 0) +
        "|ctx=" + std::to_string(llama_n_ctx(ctx)) +
        "|allowedCores=" + std::to_string(allowed_cores) +
        "|hwcap=" + std::to_string(hwcap) + "|hwcap2=" + std::to_string(hwcap2) +
        "|runtime=" + AI_CHAT_LLAMA_CPP_SHA;
}
static std::string cpu_tuning_store_path() {
    if (!g_cpu_tuning_path.empty()) return g_cpu_tuning_path;
    return g_native_trace_file.empty() ? std::string() : g_native_trace_file + ".cpu-autotune";
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeSetCpuTuningPath(
        JNIEnv *env, jclass, jstring jpath) {
    if (!jpath) { g_cpu_tuning_path.clear(); return; }
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) return;
    g_cpu_tuning_path.assign(path);
    env->ReleaseStringUTFChars(jpath, path);
    append_native_trace("NATIVE_CPU_AUTOTUNE_STORE path_configured=1 location=app_files");
}
static void load_cpu_tuning_state(const std::string &identity) {
    g_cpu_thread_tuning.clear();
    g_cpu_thread_trial_count = 0;
    const std::string path = cpu_tuning_store_path();
    std::ifstream input(path);
    uint64_t stored_key = 0;
    if (path.empty() || !(input >> stored_key) || stored_key != cpu_tuning_hash(identity)) {
        append_native_trace("NATIVE_CPU_AUTOTUNE_STORE status=miss reason=identity_or_file");
        return;
    }
    int threads = 0, attempts = 0, samples = 0;
    double ema = 0.0;
    while (input >> threads >> attempts >> samples >> ema) {
        if (threads < 1 || threads > 64 || attempts < 0 || samples < 0 ||
            !std::isfinite(ema) || ema < 0.0) continue;
        g_cpu_thread_tuning[threads] = {attempts, samples, ema};
        g_cpu_thread_trial_count += attempts;
    }
    append_native_trace((std::string("NATIVE_CPU_AUTOTUNE_STORE status=loaded configurations=") +
        std::to_string(g_cpu_thread_tuning.size()) + " trials=" +
        std::to_string(g_cpu_thread_trial_count)).c_str());
}
static void save_cpu_tuning_state(const std::string &identity) {
    const std::string path = cpu_tuning_store_path();
    if (path.empty()) return;
    const std::string temp_path = path + ".tmp";
    {
        std::ofstream output(temp_path, std::ios::trunc);
        if (!output.is_open()) {
            append_native_trace("NATIVE_CPU_AUTOTUNE_STORE status=write_failed");
            return;
        }
        output << cpu_tuning_hash(identity) << '\n';
        for (const auto &entry : g_cpu_thread_tuning) {
            output << entry.first << ' ' << entry.second.attempts << ' '
                   << entry.second.samples << ' ' << entry.second.ema_tokens_per_sec << '\n';
        }
        output.flush();
        if (!output.good()) {
            output.close();
            unlink(temp_path.c_str());
            append_native_trace("NATIVE_CPU_AUTOTUNE_STORE status=write_failed");
            return;
        }
    }
    if (rename(temp_path.c_str(), path.c_str()) != 0) {
        unlink(temp_path.c_str());
        append_native_trace("NATIVE_CPU_AUTOTUNE_STORE status=rename_failed");
        return;
    }
    append_native_trace("NATIVE_CPU_AUTOTUNE_STORE status=saved");
}

static unsigned cpu_threads_available_to_process() {
    unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0) cores = std::min(cores, static_cast<unsigned>(online));
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        const int allowed_count = CPU_COUNT(&allowed);
        if (allowed_count > 0) cores = std::min(cores, static_cast<unsigned>(allowed_count));
    }
#endif
    return std::max(1u, cores);
}
static std::vector<int> cpu_thread_candidates() {
    const unsigned cores = cpu_threads_available_to_process();
    const int max_threads = std::max(1, std::min(8, static_cast<int>(cores)));
    std::vector<int> candidates;
    for (int n = 1; n <= max_threads; ++n) candidates.push_back(n);
    return candidates;
}
static int best_measured_cpu_threads() {
    int best_threads = 0; double best_speed = -1.0;
    for (const auto &entry : g_cpu_thread_tuning) {
        if (entry.second.samples > 0 && entry.second.ema_tokens_per_sec > best_speed) {
            best_speed = entry.second.ema_tokens_per_sec; best_threads = entry.first;
        }
    }
    return best_threads;
}
static int choose_cpu_thread_trial() {
    if (!g_context || g_gpu || g_spec_requested || g_spec) return 0;
    const int current = std::max(1, (int) llama_n_threads(g_context));
    const std::string identity = cpu_tuning_identity(g_context);
    if (g_cpu_tuned_context != g_context || g_cpu_tuning_key != identity) {
        g_cpu_tuned_context = g_context;
        g_cpu_default_threads = current;
        g_cpu_tuning_key = identity;
        load_cpu_tuning_state(g_cpu_tuning_key);
    } else if (g_cpu_default_threads == 0) {
        g_cpu_default_threads = current;
    }
    const auto candidates = cpu_thread_candidates();
    const int best = best_measured_cpu_threads();

    // Never discard a valid measured winner just because the configured default
    // thread count has not yet been trialed in this process. The old ordering
    // could repeatedly pick the default (e.g. 2 threads) even when persisted
    // measurements showed a faster candidate (e.g. 4 threads).
    if (best > 0) return best;

    // No valid decode sample exists yet: collect a baseline, then move to an
    // untested candidate on subsequent generations. Do not use wall-clock or
    // prefill timings as a proxy for decode throughput.
    if (g_cpu_thread_tuning[current].attempts == 0 &&
        std::find(candidates.begin(), candidates.end(), current) != candidates.end()) return current;
    for (int candidate : candidates)
        if (g_cpu_thread_tuning[candidate].attempts == 0) return candidate;

    return g_cpu_default_threads;
}
static void record_cpu_thread_trial(int threads, int generated, int64_t decode_ms,
                                    double tokens_per_sec, int result) {
    auto &state = g_cpu_thread_tuning[threads];
    ++state.attempts;
    const bool valid_sample = result == 0 && generated >= 8 && decode_ms >= 500 &&
        std::isfinite(tokens_per_sec) && tokens_per_sec > 0.0;
    if (valid_sample) {
        state.ema_tokens_per_sec = state.samples == 0 ? tokens_per_sec
            : (0.65 * state.ema_tokens_per_sec + 0.35 * tokens_per_sec);
        ++state.samples;
    }
    ++g_cpu_thread_trial_count;
    save_cpu_tuning_state(g_cpu_tuning_key);
    const int best = best_measured_cpu_threads();
    append_native_trace((std::string("NATIVE_CPU_AUTOTUNE_RESULT trial=") +
        std::to_string(g_cpu_thread_trial_count) + " threads=" + std::to_string(threads) +
        " generatedTokens=" + std::to_string(generated) + " decodeMs=" + std::to_string(decode_ms) +
        " measuredTokensPerSec=" + std::to_string(tokens_per_sec) +
        " validSample=" + (valid_sample ? "1" : "0") +
        " samplesForThreads=" + std::to_string(state.samples) +
        " bestMeasuredThreads=" + std::to_string(best) +
        " bestMeasuredTokensPerSec=" +
        std::to_string(best > 0 ? g_cpu_thread_tuning[best].ema_tokens_per_sec : 0.0) +
        " policy=REAL_GENERATION_MEASUREMENT").c_str());
}

static void disable_speculative_runtime(const char * reason) {
    set_spec_phase("DISABLE_BEGIN");
    append_native_trace((std::string("SPECULATIVE_DISABLED reason=") + (reason ? reason : "unknown")).c_str());
    if (g_spec) {
        set_spec_phase("DISABLE_FREE_SPEC");
        common_speculative_free(g_spec);
        g_spec = nullptr;
    }
    g_spec_init.reset();
    g_spec_accept_ema = 1.0;
    set_spec_phase("DISABLE_DONE");
}

static bool init_speculative_runtime() {
    set_spec_phase("INIT_ENTER");
    append_native_trace("SPECULATIVE_INIT_ENTERED");
    if (!g_spec_requested || !g_model || !g_context) {
        set_spec_phase("INIT_NOT_READY");
        return false;
    }
    if (g_spec || g_spec_init) {
        append_native_trace("SPECULATIVE_DOUBLE_INIT_GUARD");
        disable_speculative_runtime("double_init_guard");
    }
    try {
        set_spec_phase("INIT_BUILD_PARAMS");
        common_params p;
        p.model.path = g_target_model_path;
        p.n_ctx = (int) llama_n_ctx(g_context);
        p.n_batch = 128; p.n_ubatch = 128;
        p.n_parallel = 1; p.n_sequences = 1; p.n_gpu_layers = 99;
        p.speculative.draft.n_max = 3;
        p.speculative.draft.n_gpu_layers = 99;

        // Build a true draft-model parameter set before creating the draft context.
        // common_speculative_init_from_params() derives the model loader from
        // params.model, so passing the target params here would load the target
        // GGUF again instead of the configured draft GGUF. That can double the
        // resident GPU model and crash as soon as the first speculative request
        // starts.
        common_params draft_params = common_base_params_to_speculative(p);
        // Make every draft-only field explicit on BOTH parameter objects:
        // common_speculative_init_from_params() owns the draft context, while
        // common_speculative_init() later consumes p.speculative. Keeping these
        // objects identical prevents the draft context and speculative object
        // from disagreeing about model/cache/context configuration.
        draft_params.n_ctx = (int) llama_n_ctx(g_context);
        draft_params.n_batch = 128;
        draft_params.n_ubatch = 128;
        draft_params.n_parallel = 1;
        draft_params.n_sequences = 1;
        draft_params.n_gpu_layers = 99;
        draft_params.speculative.draft.ctx_tgt = g_context;
        draft_params.speculative.draft.n_max = 3;
        draft_params.speculative.draft.n_gpu_layers = 99;

        p.speculative.draft.ctx_tgt = g_context;
        p.speculative.draft.n_max = 3;
        p.speculative.draft.n_gpu_layers = 99;

        if (g_spec_mtp) {
            p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
            set_spec_phase("INIT_FROM_PARAMS");
            append_native_trace("SPECULATIVE_INIT_FROM_PARAMS_BEGIN");
            g_spec_init = common_speculative_init_from_params(draft_params, g_model, g_context);
            append_native_trace("SPECULATIVE_INIT_FROM_PARAMS_END");
            if (!g_spec_init || !g_spec_init->context()) {
                g_spec_init.reset();
                append_native_trace("MTP_INIT_CONTEXT_FAILED");
                return false;
            }
            p.speculative.draft.ctx_dft = g_spec_init->context();
        } else {
            p.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
            draft_params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
            draft_params.model.path = g_spec_draft_path;
            p.speculative.draft.mparams.path = g_spec_draft_path;
            // draft_params.model.path and mparams.path are intentionally both
            // set above; keep the speculative object synchronized as well.
            append_native_trace((std::string("SPECULATIVE_DRAFT_PARAM_PATH=") + g_spec_draft_path).c_str());
            g_spec_init = common_speculative_init_from_params(draft_params, g_model, g_context);
            if (!g_spec_init || !g_spec_init->context() || !g_spec_init->model()) {
                g_spec_init.reset();
                return false;
            }
            if (!common_speculative_are_compatible(g_model, g_spec_init->model())) {
                append_native_trace("SPECULATIVE_INCOMPATIBLE_VOCAB");
                g_spec_init.reset();
                return false;
            }
            p.speculative.draft.ctx_dft = g_spec_init->context();
        }

        set_spec_phase("INIT_SPEC_OBJECT");
        append_native_trace("SPECULATIVE_OBJECT_INIT_BEGIN");
        g_spec = common_speculative_init(p.speculative, 1);
        append_native_trace("SPECULATIVE_OBJECT_INIT_END");
        if (!g_spec) {
            append_native_trace("SPECULATIVE_OBJECT_INIT_FAILED");
            g_spec_init.reset();
            set_spec_phase("INIT_SPEC_OBJECT_FAILED");
            return false;
        }
        set_spec_phase("READY");
        append_native_trace((std::string("SPECULATIVE_READY mode=") +
            (g_spec_mtp ? "MTP" : "DRAFT") + " nMax=3").c_str());
        return true;
    } catch (const std::exception &e) {
        append_native_trace((std::string("SPECULATIVE_EXCEPTION ") + e.what()).c_str());
        disable_speculative_runtime("exception");
        return false;
    }
}
static common_params_sampling spec_sampling(float temperature,int top_k,float top_p,float min_p){
    common_params_sampling p; p.top_k=std::max(0,top_k); p.top_p=std::clamp(top_p,0.f,1.f); p.min_p=std::clamp(min_p,0.f,1.f);
    p.temp=std::max(0.f,temperature); p.penalty_last_n=64; p.penalty_repeat=1.1f;
    p.samplers={COMMON_SAMPLER_TYPE_PENALTIES,COMMON_SAMPLER_TYPE_TOP_K,COMMON_SAMPLER_TYPE_TOP_P,COMMON_SAMPLER_TYPE_MIN_P,COMMON_SAMPLER_TYPE_TEMPERATURE};
    return p;
}

static bool emit_spec_token(JNIEnv *env, jobject listener, jmethodID on_token, const llama_vocab *vocab, llama_token tok) {
    char piece[1024];
    const int z = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
    if (z <= 0) return true;
    jstring out = env->NewStringUTF(std::string(piece, (size_t) z).c_str());
    if (!out) return false;
    env->CallVoidMethod(listener, on_token, out);
    env->DeleteLocalRef(out);
    if (env->ExceptionCheck()) { env->ExceptionClear(); return false; }
    return true;
}

static int generate_spec(JNIEnv *env, jobject listener, jmethodID on_token, const std::vector<llama_token>& inp, int max_tokens, float temperature, int top_k, float top_p, float min_p) {
    int draft_tokens = 0, accepted_tokens = 0, steps = 0;
    const auto speculative_started = std::chrono::steady_clock::now();
    auto finish = [&]() {
        append_speculative_stats_trace(draft_tokens, accepted_tokens, steps);
        const auto finished = std::chrono::steady_clock::now();
        const auto ms = elapsed_ms(speculative_started, finished);
        const double tps = ms > 0 ? (1000.0 * static_cast<double>(accepted_tokens) / static_cast<double>(ms)) : 0.0;
        append_native_trace((std::string("SPECULATIVE_PERFORMANCE generationMs=") + std::to_string(ms) +
            " tokensPerSec=" + std::to_string(tps)).c_str());
    };

    set_spec_phase("GENERATE_ENTER");
    append_native_trace((std::string("SPECULATIVE_GENERATE_ENTER promptTokens=") +
        std::to_string(inp.size()) + " maxTokens=" + std::to_string(max_tokens)).c_str());

    llama_context * dctx = g_spec_init ? g_spec_init->context() : nullptr;
    if (!g_spec || !dctx || inp.size() < 2 || !on_token) {
        set_spec_phase("GENERATE_INVALID_STATE");
        finish();
        return -1;
    }

    llama_memory_clear(llama_get_memory(g_context), true);
    llama_memory_clear(llama_get_memory(dctx), true);

    common_params_sampling sp = spec_sampling(temperature, top_k, top_p, min_p);
    common_sampler_ptr smp(common_sampler_init(g_model, sp));
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (!smp || !vocab) {
        finish();
        return -1;
    }

    const int n_prompt = (int) inp.size();
    const int cap = std::max(1, (int) llama_n_batch(g_context));
    if (n_prompt - 1 > cap) {
        append_native_trace((std::string("SPECULATIVE_PROMPT_BATCH_LIMIT promptMinusLast=") +
            std::to_string(n_prompt - 1) + " nBatch=" + std::to_string(cap)).c_str());
        finish();
        return 10;
    }

    llama_batch batch = llama_batch_init(cap + 1, 0, 1);
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) {
        llama_batch_free(batch);
        finish();
        return -1;
    }

    auto fill = [&](const llama_tokens & tokens, int pos, bool last_logits) {
        batch.n_tokens = (int) tokens.size();
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[(size_t)i];
            batch.pos[i] = (llama_pos)(pos + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (last_logits && i == batch.n_tokens - 1) ? 1 : 0;
        }
    };

    const llama_seq_id seq_id = 0;

    // common_context_can_seq_rm() clears the context while probing its memory
    // capabilities in this pinned llama.cpp revision. Therefore it MUST happen
    // before target prefill, not after common_speculative_begin().
    const common_context_seq_rm_type tgt_rm = common_context_can_seq_rm(g_context);
    const common_context_seq_rm_type dft_rm = common_context_can_seq_rm(dctx);
    const bool use_ckpt_tgt = tgt_rm == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = dft_rm == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    append_native_trace((std::string("SPEC_CONTEXT_ROLLBACK target=") + std::to_string((int)tgt_rm) +
        " draft=" + std::to_string((int)dft_rm) +
        " targetCheckpoint=" + (use_ckpt_tgt ? "1" : "0") +
        " draftCheckpoint=" + (use_ckpt_dft ? "1" : "0")).c_str());

    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(g_context));
    llama_token id_last = inp.back();
    int n_past = (int) prompt_tgt.size();
    int n_predict = 0;

    // This is the exact lifecycle used by the pinned llama.cpp speculative example:
    // Target prefill -> common_speculative_process(prompt) -> common_speculative_begin().
    set_spec_phase("SPECULATIVE_PREFILL_BEGIN");
    append_native_trace("SPECULATIVE_PREFILL_BEGIN");

    {
        common_batch batch_prompt(g_context);
        for (size_t i = 0; i < prompt_tgt.size(); ++i) {
            batch_prompt.add(prompt_tgt[i], (llama_pos)i, seq_id, false);
        }

        fill(prompt_tgt, 0, false);
        if (llama_decode(g_context, batch) != 0) {
            append_native_trace("SPECULATIVE_TARGET_PREFILL_FAILED");
            llama_batch_free(batch);
            finish();
            return 10;
        }

        if (!common_speculative_process(g_spec, batch_prompt)) {
            append_native_trace("SPECULATIVE_PREFILL_PROCESS_FAILED");
            llama_batch_free(batch);
            finish();
            return 10;
        }
    }

    set_spec_phase("GENERATE_BEGIN_SPECULATIVE");
    common_speculative_begin(g_spec, seq_id, prompt_tgt);
    append_native_trace("SPEC_BEGIN_DONE");

    common_batch batch_tgt(g_context);
    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    while (n_predict < max_tokens && !g_stop.load(std::memory_order_relaxed)) {
        if (draft.empty()) {
            ckpt.update_pos(
                prompt_tgt.size(),
                llama_memory_seq_pos_min(llama_get_memory(g_context), seq_id),
                llama_memory_seq_pos_max(llama_get_memory(g_context), seq_id)
            );

            if (use_ckpt_dft) {
                ckpt.update_dft(dctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            int n_draft_max = (int) llama_n_ctx(g_context) - n_past - 2;
            n_draft_max = std::min(n_draft_max, max_tokens - n_predict - 1);
            n_draft_max = std::max(n_draft_max, 0);

            common_speculative_get_draft_params(g_spec, seq_id) = {
                true,
                n_draft_max,
                n_past,
                id_last,
                &prompt_tgt,
                &draft,
                nullptr,
                temperature,
                LLAMA_DEFAULT_SEED,
            };

            set_spec_phase("DRAFT_GENERATE_BEGIN");
            append_native_trace((std::string("SPEC_DRAFT_GENERATE_BEGIN pos=") +
                std::to_string(n_past) + " nMax=" + std::to_string(n_draft_max)).c_str());

            common_speculative_draft(g_spec);

            append_native_trace((std::string("SPEC_DRAFT_GENERATE_END tokens=") +
                std::to_string(draft.size())).c_str());
            draft_tokens += (int) draft.size();

            if (!draft.empty() && use_ckpt_tgt) {
                ckpt.update_tgt(g_context, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // The draft model has already advanced through its proposal. Restore
            // the checkpoint before target verification, exactly as upstream does.
            if (use_ckpt_dft) {
                ckpt.load_dft(dctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            if (dctx) {
                const llama_pos rm_from = ckpt.pos_max + 1;
                if (!llama_memory_seq_rm(llama_get_memory(dctx), seq_id, rm_from, -1)) {
                    append_native_trace((std::string("SPEC_DRAFT_ROLLBACK_FAILED from=") +
                        std::to_string(rm_from)).c_str());
                    llama_batch_free(batch);
                    finish();
                    return 20;
                }
            }
        } else if (use_ckpt_tgt && ckpt.empty()) {
            append_native_trace("SPEC_CHECKPOINT_MISSING_FOR_PARTIAL_DRAFT");
            llama_batch_free(batch);
            finish();
            return 20;
        }

        // Verify [id_last, draft...] on the target.
        batch_tgt.clear();
        batch_tgt.add(id_last, n_past++, seq_id, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            batch_tgt.add(draft[i], n_past + (int)i, seq_id, true);
        }

        const int verify_count = batch_tgt.size();
        append_native_trace((std::string("SPEC_VERIFY_BEGIN targetStart=") +
            std::to_string(n_past - 1) + " verifyCount=" + std::to_string(verify_count) +
            " draftCount=" + std::to_string(draft.size())).c_str());

        if (llama_process(g_context, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get()) != 0) {
            append_native_trace("SPEC_VERIFY_TARGET_DECODE_FAILED");
            llama_batch_free(batch);
            finish();
            return 10;
        }

        // The speculative implementation must see the exact target verification
        // batch before accept(). This drives the draft/MTP implementation state.
        if (!common_speculative_process(g_spec, batch_tgt)) {
            append_native_trace("SPEC_VERIFY_SPECULATIVE_PROCESS_FAILED");
            llama_batch_free(batch);
            finish();
            return 10;
        }
        append_native_trace("SPEC_VERIFY_SPECULATIVE_PROCESS_DONE");

        common_sampler_ptr smp_save;
        if (use_ckpt_tgt) {
            smp_save.reset(common_sampler_clone(smp.get()));
        }

        const size_t n_draft = draft.size();
        set_spec_phase("VERIFY_SAMPLE_ACCEPT");
        auto ids = common_sampler_sample_and_accept_n(smp.get(), g_context, draft);
        append_native_trace((std::string("SPEC_SAMPLE_ACCEPT_END ids=") +
            std::to_string(ids.size())).c_str());

        if (ids.empty()) {
            append_native_trace("SPEC_SAMPLE_ACCEPT_EMPTY");
            llama_batch_free(batch);
            finish();
            return 10;
        }

        const int accepted = std::max(0, (int)ids.size() - 1);
        accepted_tokens += accepted;
        ++steps;

        if (n_draft > 0) {
            const double round_rate = static_cast<double>(accepted) / static_cast<double>(n_draft);
            g_spec_accept_ema = 0.75 * g_spec_accept_ema + 0.25 * round_rate;
            append_native_trace((std::string("SPECULATIVE_ADAPT acceptance=") +
                std::to_string(round_rate) + " ema=" + std::to_string(g_spec_accept_ema)).c_str());
        }

        // Hybrid/recurrent target rollback must restore the exact state before
        // the speculative proposal when the target accepted only part of it.
        if (use_ckpt_tgt && accepted < (int)n_draft) {
            append_native_trace((std::string("SPEC_PARTIAL_ACCEPTANCE accepted=") +
                std::to_string(accepted) + " drafted=" + std::to_string(n_draft) +
                " action=RESTORE_CHECKPOINT").c_str());

            draft = std::move(ids);
            ckpt.load_tgt(g_context, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (!llama_memory_seq_rm(llama_get_memory(g_context), seq_id, ckpt.pos_max + 1, -1)) {
                append_native_trace("SPEC_TARGET_CHECKPOINT_ROLLBACK_FAILED");
                llama_batch_free(batch);
                finish();
                return 20;
            }

            if (dctx) {
                ckpt.load_dft(dctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (!llama_memory_seq_rm(llama_get_memory(dctx), seq_id, ckpt.pos_max + 1, -1)) {
                    append_native_trace("SPEC_DRAFT_CHECKPOINT_ROLLBACK_FAILED");
                    llama_batch_free(batch);
                    finish();
                    return 20;
                }
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smp = std::move(smp_save);
            n_past = (int) prompt_tgt.size();
            continue;
        }

        set_spec_phase("ACCEPT_DRAFT");
        common_speculative_accept(g_spec, seq_id, (uint16_t)accepted);
        append_native_trace((std::string("SPEC_ACCEPT_DONE accepted=") + std::to_string(accepted)).c_str());

        n_past += accepted;
        n_predict += (int)ids.size();

        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);
            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                llama_batch_free(batch);
                finish();
                return 0;
            }

            if (!emit_spec_token(env, listener, on_token, vocab, id_last)) {
                llama_batch_free(batch);
                finish();
                return 12;
            }
        }

        draft.clear();

        // Discard any unaccepted target/draft memory. For recurrent/hybrid
        // contexts this call is only used where the backend explicitly supports
        // the requested operation; otherwise the checkpoint path above handled
        // partial rollback.
        if (!llama_memory_seq_rm(llama_get_memory(g_context), seq_id, n_past, -1)) {
            append_native_trace((std::string("SPEC_TARGET_TAIL_CLEAR_FAILED pos=") +
                std::to_string(n_past)).c_str());
            llama_batch_free(batch);
            finish();
            return 20;
        }
        if (dctx && !llama_memory_seq_rm(llama_get_memory(dctx), seq_id, n_past, -1)) {
            append_native_trace((std::string("SPEC_DRAFT_TAIL_CLEAR_FAILED pos=") +
                std::to_string(n_past)).c_str());
            llama_batch_free(batch);
            finish();
            return 20;
        }
    }

    llama_batch_free(batch);
    finish();
    return g_stop.load(std::memory_order_relaxed) ? 9 : 0;
}

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
        jobject listener) {
    std::lock_guard<std::mutex> runtime_lock(g_native_runtime_mutex);
    checkpoint("NATIVE_GENERATE_ENTERED");
    if (!g_context || !g_model) { checkpoint("NATIVE_GENERATE_NO_MODEL"); return 2; }
    int cpu_thread_trial = 0;
    if (!g_gpu && !g_spec_requested && !g_spec) {
        cpu_thread_trial = choose_cpu_thread_trial();
        if (cpu_thread_trial > 0) {
            // Decode throughput is the score being optimized. Keep the separately
            // configured batch/prefill worker count fixed so prompt length and
            // prefill scheduling do not masquerade as a decode-thread improvement.
            const int batch_threads = std::max(1, (int) llama_n_threads_batch(g_context));
            llama_set_n_threads(g_context, cpu_thread_trial, batch_threads);
            append_native_trace((std::string("NATIVE_CPU_AUTOTUNE_SELECTED decodeThreads=") +
                std::to_string(cpu_thread_trial) + " batchThreads=" +
                std::to_string(batch_threads) + " allowedCpuThreads=" +
                std::to_string(cpu_threads_available_to_process()) + " trial=" +
                std::to_string(g_cpu_thread_trial_count + 1) + " candidateCount=" +
                std::to_string(cpu_thread_candidates().size()) +
                " mode=" + (best_measured_cpu_threads() > 0 ? "explore_then_best" : "exploration") +
                " reason=CPU_FALLBACK").c_str());
        }
    }
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
    if (g_spec) {
        jclass spec_listener_class = env->GetObjectClass(listener);
        jmethodID spec_on_token = spec_listener_class ? env->GetMethodID(spec_listener_class, "onToken", "(Ljava/lang/String;)V") : nullptr;
        if (spec_on_token) {
            std::vector<llama_token> spec_tokens((size_t)n_prompt);
            if (llama_tokenize(vocab,prompt_text.c_str(),prompt_text.size(),spec_tokens.data(),spec_tokens.size(),true,true)<0) {
                if (spec_listener_class) env->DeleteLocalRef(spec_listener_class);
                return 5;
            }
            const int spec_capacity = std::max(1, (int)llama_n_ctx(g_context) - n_prompt - 1);
            const int sr=generate_spec(env,listener,spec_on_token,spec_tokens,std::min((int)max_tokens,spec_capacity),(float)temperature,(int)top_k,(float)top_p,(float)min_p);
            append_native_trace((std::string("SPECULATIVE_GENERATE_RETURNED code=")+std::to_string(sr)).c_str());
            if (spec_listener_class) env->DeleteLocalRef(spec_listener_class);
            if(sr==0||sr==9)return sr;
            append_native_trace("SPECULATIVE_FALLBACK_TO_TARGET");
            disable_speculative_runtime("generation_failure");
            llama_memory_clear(llama_get_memory(g_context),true);
        } else if (spec_listener_class) {
            env->DeleteLocalRef(spec_listener_class);
        }
    }
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
        g_cached_prompt_state.clear();
        g_cached_prompt_state_on_device = false;
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
    // Attention-only models can trim the resident KV cache by sequence position.
    // Hybrid/recurrent models cannot: their recurrent state is path-dependent.
    // For those models we reuse only an exact cached prompt state captured
    // immediately after prompt prefill and before generation.
    // Hybrid/recurrent memory rollback is deliberately disabled for now.
// A resident recurrent state cannot be reconstructed safely from seq positions,
// and the previous exact-prefix reuse path could carry state across turns that
// looked compatible but was not semantically identical. Attention-only KV reuse
// remains enabled; hybrid models always rebuild from the complete bounded prompt.
// This is a correctness/stability guard, not a CPU fallback.
const bool cache_supported = !llama_model_has_encoder(g_model) && !hybrid_memory;
if (hybrid_memory) {
    append_native_trace("NATIVE_KV_CACHE_HYBRID_DISABLED reason=stability_guard");
}
    const llama_pos resident_min = llama_memory_seq_pos_min(memory, 0);
    const llama_pos resident_max = llama_memory_seq_pos_max(memory, 0);
    // Recurrent/hybrid models expose only their latest recurrent sequence
    // position through seq_pos_min/max; that is NOT a valid representation of
    // the full resident state. Qwen3.8 is hybrid, so validating
    // (resident_max + 1) against the full conversation token count incorrectly
    // invalidated every completed cache publish and caused every next turn to
    // become a MISS.
    const bool resident_sequence_matches_cache = hybrid_memory
            ? (!g_cached_prompt_tokens.empty() &&
               resident_max >= 0 &&
               static_cast<size_t>(resident_max + 1) == g_cached_prompt_tokens.size())
            : (!g_cached_prompt_tokens.empty() &&
               resident_min == 0 &&
               resident_max >= 0 &&
               static_cast<size_t>(resident_max + 1) == g_cached_prompt_tokens.size());
    bool cache_reused = cache_supported && resident_sequence_matches_cache && common_prefix > 0;
    size_t reuse_prefix = 0;
    const bool exact_cached_prefix = cache_reused &&
            common_prefix == g_cached_prompt_tokens.size();
    append_native_trace((std::string("NATIVE_KV_CACHE_PREFIX_CHECK supported=") +
        (cache_supported ? "1" : "0") +
        " hybrid=" + (hybrid_memory ? "1" : "0") +
        " cachedTokens=" + std::to_string(g_cached_prompt_tokens.size()) +
        " commonPrefix=" + std::to_string(common_prefix) +
        " stateBytes=" + std::to_string(g_cached_prompt_state.size()) +
        " residentMin=" + std::to_string(resident_min) +
        " residentMax=" + std::to_string(resident_max) +
        " residentMatches=" + (resident_sequence_matches_cache ? "1" : "0")).c_str());

    if (hybrid_memory) {
        // Hybrid/recurrent contexts must be rebuilt from the bounded prompt.
        // Never append a new turn to a post-generation recurrent state.
        cache_reused = false;
        reuse_prefix = 0;
        g_cached_prompt_state.clear();
        g_cached_prompt_tokens.clear();
        llama_memory_clear(memory, true);
        checkpoint("NATIVE_KV_CACHE_HYBRID_REBUILD");
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

    const size_t diagnostic_cached_tokens = g_cached_prompt_tokens.size();
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
    if (cpu_thread_trial > 0) {
        const double cpu_decode_tokens_per_sec = generated > 0 && g_decode_profile.decode_ms > 0
            ? (1000.0 * static_cast<double>(generated) /
               static_cast<double>(g_decode_profile.decode_ms))
            : 0.0;
        record_cpu_thread_trial(cpu_thread_trial, generated, g_decode_profile.decode_ms,
                               cpu_decode_tokens_per_sec, result);
    }
    const auto profile_prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_finished_at - generation_started_at).count();
    append_decode_profile_trace(generation_ms, profile_prefill_ms);
    append_native_trace((std::string("NATIVE_EXECUTION_PROFILE mode=") +
        (g_gpu && g_gpu_backend_loaded ? "OPENCL_GPU_PREFERRED_MIXED_CPU_OPS" : "CPU_FALLBACK") +
        " gpuBackendRegistered=" + (g_gpu_backend_loaded ? "1" : "0") +
        " fallbackPolicy=ggml_scheduler" +
        " decodeSteps=" + std::to_string(g_decode_profile.token_steps) +
        " decodeWallMs=" + std::to_string(g_decode_profile.decode_ms) +
        " cpuLogitsSyncMs=" + std::to_string(g_decode_profile.logits_sync_ms) +
        " cpuSamplingMs=" + std::to_string(g_decode_profile.sampling_ms) +
        " cpuCallbackMs=" + std::to_string(g_decode_profile.callback_ms) +
        " cpuMeasuredSubtotalMs=" + std::to_string(g_decode_profile.logits_sync_ms +
            g_decode_profile.sampling_ms + g_decode_profile.callback_ms) +
        " gpuKernelTiming=cl_profiling.csv").c_str());
    if (prefill_recorded && (result == 0 || result == 9)) {
        // Publish only the token sequence that is actually resident in the live
        // context. The next request can then reuse the largest exact prefix.
        if (!hybrid_memory) {
            g_cached_prompt_tokens = prompt_tokens;
            g_cached_prompt_tokens.insert(
                    g_cached_prompt_tokens.end(),
                    generated_decoded_tokens.begin(),
                    generated_decoded_tokens.end());
        } else {
            g_cached_prompt_tokens.clear();
        }
        g_cached_prompt_state.clear();
        g_cached_prompt_state_on_device = false;
        g_cached_model = g_model;
        g_cached_context = g_context;
        const llama_pos published_min = llama_memory_seq_pos_min(memory, 0);
        const llama_pos published_max = llama_memory_seq_pos_max(memory, 0);
        const bool resident_publish_valid = !hybrid_memory &&
                published_min == 0 &&
                published_max >= 0 &&
                static_cast<size_t>(published_max + 1) == g_cached_prompt_tokens.size();
        if (!resident_publish_valid) {
            append_native_trace((std::string("NATIVE_KV_CACHE_PUBLISH_INVALID residentMin=") +
                std::to_string(published_min) + " residentMax=" +
                std::to_string(published_max) + " cachedTokens=" +
                std::to_string(g_cached_prompt_tokens.size()) +
                " hybrid=" + (hybrid_memory ? "1" : "0")).c_str());
            g_cached_prompt_tokens.clear();
            g_cached_model = nullptr;
            g_cached_context = nullptr;
            llama_memory_clear(memory, true);
        } else {
            append_native_trace((std::string("NATIVE_KV_CACHE_PUBLISHED tokens=") +
                std::to_string(g_cached_prompt_tokens.size()) +
                " promptTokens=" + std::to_string(prompt_tokens.size()) +
                " generatedDecodedTokens=" + std::to_string(generated_decoded_tokens.size()) +
                " residentMin=" + std::to_string(published_min) +
                " residentMax=" + std::to_string(published_max) +
                " mode=" + (hybrid_memory ? "resident_hybrid" : "resident_kv")).c_str());
        }
    } else if (prefill_recorded) {
        g_cached_prompt_tokens.clear();
        g_cached_prompt_state.clear();
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
