#include "llama.h"
#include <signal.h>
#include <unistd.h>

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
