# Chat and voice UI boundaries

## Separate interaction surfaces

- The ordinary text conversation remains hosted by `MainActivity` / `MainScreen` and its `ChatPage` UI.
- The full voice conversation is hosted by the dedicated `VoiceConversationActivity`, which owns its own Activity and Compose lifecycle and renders `VoiceConversationHost` / `VoiceConversationPage`.
- Opening or closing the voice screen no longer switches the text-chat destination inside `MainScreen`; the text-chat state remains owned by the text-chat screen.
- Voice-only settings embedded in the main Settings screen are settings content, not the active voice conversation surface.

## Intentionally shared application services

Both surfaces continue to use the same application-scoped `AppContainer` and therefore share:
- The active model/runtime and provider selection.
- Model activation/management.
- Conversation history storage.
- Runtime diagnostics and observability.
- The screen-awareness context provider where the current provider configuration allows it.

These are shared services, not shared UI state. They must not be duplicated just to separate screens. Both modes use the shared session/runtime foundation, but they build different requests and own different UI state.

## Voice-only ownership

The voice Activity owns the voice conversation host and its controller lifecycle. Voice turns explicitly create an AgentSession with `includeAgentTools = false`; the text-chat Agent tool catalog and action protocol are not injected into voice prompts. Voice also supplies its own bounded inference settings (384 new tokens, 12 recent messages, Thinking disabled), rather than inheriting the text-chat settings reader. STT/TTS model selection, microphone capture, speech segmentation, playback, wake-trigger handoff, and voice transcript UI remain in the voice feature. Disposing the voice Activity disposes the voice host/controller without recreating the text-chat screen.

## Voice shutdown and thread ownership

VoiceConversationController.stop() must not block the Activity/UI caller while AudioRecord.stop() or the microphone capture-thread join is running. Microphone shutdown is dispatched to Dispatchers.IO; LocalMicrophone.start() and stop() are synchronized so a concurrent start/stop cannot release the same recorder twice. Voice engine implementations are separated by responsibility (LocalPiperTts, LocalAsrEngines, LocalMicrophone) while remaining in the same package and preserving their existing contracts.

## Text-chat ownership

The text chat retains its own composer, message stream, Agent execution state, approval flow, and ChatPage presentation in MainScreen. Changes to the voice transcript layout or voice controls should be confined to the voice feature unless a deliberately shared service contract changes.

## Floating overlay lifecycle

The floating overlay service is a lifecycle-aware host for its Compose popup. The popup receives the service lifecycle owner before composition, and opening failures are logged and surfaced without crashing or silently stopping the bubble service.

## Home-screen chat widget

The compact RemoteViews preview wraps long unbroken tokens and opens a separate scrollable reply reader when tapped. The full answer is retained separately from the bounded preview.


## Streaming boundary

- Text-chat sessions explicitly opt into `textChatStreaming` from `AppContainer.createAgentSession(includeAgentTools = true)`.
- Text chat forwards ordinary answer chunks as soon as the stream can be distinguished from a partial `<think>` marker. Reasoning blocks are filtered incrementally, and structured action payloads are withheld from the chat transcript.
- Voice sessions explicitly pass `includeAgentTools = false`, which keeps `textChatStreaming` disabled. Their existing `AgentEvent.Token` handling, partial transcript updates, speech segmentation, and TTS playback path remain unchanged.
- Do not move text-chat stream filtering into the shared runtime callback or the voice controller. Any future stream changes must preserve this explicit mode boundary.
