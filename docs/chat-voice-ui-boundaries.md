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

These are shared services, not shared UI state. They must not be duplicated just to separate screens.

## Voice-only ownership

The voice Activity owns the voice conversation host and its controller lifecycle. STT/TTS model selection, microphone capture, speech segmentation, playback, wake-trigger handoff, and voice transcript UI remain in the voice feature. Disposing the voice Activity disposes the voice host/controller without recreating the text-chat screen.

## Text-chat ownership

The text chat retains its own composer, message stream, Agent execution state, approval flow, and ChatPage presentation in MainScreen. Changes to the voice transcript layout or voice controls should be confined to the voice feature unless a deliberately shared service contract changes.

## Floating overlay lifecycle

The floating overlay service is a lifecycle-aware host for its Compose popup. The popup receives the service lifecycle owner before composition, and opening failures are logged and surfaced without crashing or silently stopping the bubble service.

## Home-screen chat widget

The compact RemoteViews preview wraps long unbroken tokens and opens a separate scrollable reply reader when tapped. The full answer is retained separately from the bounded preview.
