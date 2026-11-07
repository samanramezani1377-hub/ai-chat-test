#ifndef NEEDLE_H
#define NEEDLE_H

#ifndef NEEDLE_API
#define NEEDLE_API __attribute__((visibility("default")))
#endif

#define NEEDLE_TEXT 1
#define NEEDLE_SPEECH 2

#ifdef __cplusplus
extern "C" {
#endif

NEEDLE_API int needle_load(const unsigned char* cact, unsigned long long n);
NEEDLE_API int needle_models(void);
NEEDLE_API const char* needle_last_error(void);
NEEDLE_API int needle_init(const char* system_prompt, const char* tools_json, const char* tool_index_path);
NEEDLE_API int needle_complete(const char* input, const float* pcm, int samples,
                               int max_new_tokens, char* out, int out_capacity);
NEEDLE_API void needle_set_audio(const char* language, const char* keywords, int word_timestamps);
NEEDLE_API void needle_reset(void);
NEEDLE_API int needle_transcribe(const float* pcm, int samples, const char* language,
                                 const char* keywords, int word_timestamps,
                                 char* out, int out_capacity);
NEEDLE_API int needle_stream_transcribe_process(const float* pcm, int samples,
                                                const char* language, const char* keywords,
                                                char* out, int out_capacity);
NEEDLE_API int needle_stream_transcribe_stop(char* out, int out_capacity);
NEEDLE_API int needle_embed(const char* input, const float* pcm, int samples,
                            float* out, int out_capacity);

#ifdef __cplusplus
}
#endif
#endif
