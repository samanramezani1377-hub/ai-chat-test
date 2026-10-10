package com.woogit.aicore.runtime.mnn
internal object MnnNative {
    init { System.loadLibrary("ai_chat_mnn_runtime") }
    external fun nativeLoad(configPath: String, cpuThreads: Int): String
    external fun nativeGenerate(roles: Array<String>, contents: Array<String>, maxTokens: Int): String
    external fun nativeStop()
    external fun nativeUnload()
    external fun nativeBackend(): String
}
