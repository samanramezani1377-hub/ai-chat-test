# درخت مرجع رابط کاربری — UI v2

**وضعیت:** مرجع جاری ساختار UI
**جزئیات:** [REDESIGN_V2.md](REDESIGN_V2.md)

~~~text
برنامه
├── App Shell
│   ├── Header / Runtime Status
│   ├── Sidebar
│   └── Content Host
├── CHAT (Default)
│   ├── Conversation Messages
│   ├── Streaming Assistant
│   ├── Agent / Action State
│   ├── Approval Controls
│   └── Composer
├── HISTORY
│   ├── Search (title + loaded message text)
│   ├── Preview / Last Activity
│   ├── Open / Rename / Delete / Undo
│   └── Load More
├── SETTINGS
│   ├── AI / Models / Provider
│   ├── Voice / Shenava / Piper / Laya / Dina
│   ├── Inference / Sampling
│   ├── Context
│   └── Agent
├── VOICE
│   ├── Live Transcript
│   ├── Streaming Assistant Text
│   ├── Microphone / Listening State
│   ├── TTS Playback / Interrupt
│   └── Stop Conversation
├── WORKSPACE
├── DIAGNOSTICS
└── ABOUT
~~~

## قراردادهای اصلی

- ورود عادی: CHAT.
- Wake Word: ورود مستقیم به VOICE در صورت درخواست سرویس.
- History و Chat از یک ConversationHistoryRepository استفاده می‌کنند.
- تنظیمات Voice از داخل Settings قابل دسترسی است؛ Voice screen برای مکالمه است.
- تمام تنظیمات مدل و Provider از Settings مدیریت می‌شوند.
- Local CPU و CPU fallback مسیرهای معتبر هستند؛ GPU شتاب‌دهنده اختیاری است.
- منبع Inference بین Local و Provider بدون انتخاب کاربر تغییر نمی‌کند.
- تمام متن‌های UI فارسی و RTL هستند؛ داده‌های فنی می‌توانند LTR باشند.
- هیچ UI نباید قابلیت یا وضعیت فنی را جعل کند.
~~~
