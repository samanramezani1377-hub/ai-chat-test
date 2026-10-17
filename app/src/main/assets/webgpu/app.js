import {
  AutoProcessor,
  Qwen3_5ForConditionalGeneration,
  TextStreamer,
  InterruptableStoppingCriteria,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "onnx-community/Qwen3.5-2B-ONNX-OPT";
const $ = (id) => document.getElementById(id);
const status = $("status");
const detail = $("detail");
const loadButton = $("load");
const resetButton = $("reset");
const messages = $("messages");
const composer = $("composer");
const input = $("input");
const sendButton = $("send");
const stopButton = $("stop");
let processor = null;
let model = null;
let generating = false;
let history = [];
let stoppingCriteria = new InterruptableStoppingCriteria();

env.useBrowserCache = true;
env.useWasmCache = true;
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "https://appassets.androidplatform.net/models/";

function setStatus(title, description) {
  status.textContent = title;
  detail.textContent = description;
}
function addMessage(role, text) {
  const welcome = messages.querySelector(".welcome");
  if (welcome) welcome.remove();
  const node = document.createElement("div");
  node.className = "message " + role;
  node.textContent = text;
  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}
function showError(error) {
  const text = error?.message || String(error);
  addMessage("assistant", "خطا: " + text);
  setStatus("اجرای WebGPU ناموفق بود", text);
}

async function loadLocalModel() {
  setStatus("در حال بارگذاری مدل محلی…", "فایل‌ها از پوشهٔ پایدار دینا خوانده می‌شوند؛ دانلود مجدد انجام نمی‌شود.");
  processor = await AutoProcessor.from_pretrained(MODEL_ID, { local_files_only: true });
  model = await Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
    dtype: { embed_tokens: "q4", decoder_model_merged: "q4" },
    device: "webgpu",
    local_files_only: true,
  });
  setStatus("مدل آماده است", "WebGPU فعال است؛ مدل از حافظهٔ محلی دینا بارگذاری شد.");
  addMessage("assistant", "سلام! من دینا هستم. مدل Qwen3.5-2B از حافظهٔ محلی بارگذاری شد. چه کمکی از دستم برمی‌آید؟");
  input.disabled = false;
  resetButton.disabled = false;
  input.focus();
}

window.dinaModelProgress = async (state, message) => {
  if (state === "progress") {
    setStatus("در حال دریافت مدل…", message);
  } else if (state === "done") {
    setStatus("فایل‌های مدل آماده‌اند", message);
    try {
      await loadLocalModel();
    } catch (error) {
      console.error(error);
      showError(error);
      loadButton.disabled = false;
    }
  } else if (state === "error") {
    showError(new Error(message));
    loadButton.disabled = false;
  }
};

loadButton.addEventListener("click", async () => {
  loadButton.disabled = true;
  setStatus("در حال بررسی قابلیت‌ها…", "بررسی Secure Context و WebGPU مرورگر داخلی اندروید.");
  try {
    if (!window.isSecureContext) throw new Error("این صفحه Secure Context نیست؛ WebGPU در این مبدأ امن در دسترس نیست.");
    if (!navigator.gpu) throw new Error("WebGPU در WebView این دستگاه در دسترس نیست. Chrome/WebView را به‌روز کن یا از گفت‌وگوی محلی فعلی استفاده کن.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("مرورگر WebGPU دارد اما آداپتور GPU قابل استفاده پیدا نشد.");
    const info = adapter.info || {};
    const gpuName = [info.vendor, info.architecture, info.device].filter(Boolean).join(" · ") || "GPU adapter (جزئیات دستگاه مخفی است)";
    setStatus("GPU شناسایی شد", gpuName);
    if (!window.DinaModelStore) throw new Error("اتصال ذخیره‌سازی محلی مدل در دسترس نیست.");
    if (window.DinaModelStore.isModelReady()) await loadLocalModel();
    else {
      setStatus("مدل روی گوشی نیست", "دانلود فایل‌های Q4 شروع می‌شود و در پوشهٔ مدل‌های دینا کنار Laya نگه‌داری خواهد شد.");
      window.DinaModelStore.startModelDownload();
    }
  } catch (error) {
    console.error(error);
    showError(error);
    loadButton.disabled = false;
  }
});

input.addEventListener("input", () => {
  sendButton.disabled = !model || generating || !input.value.trim();
});
resetButton.addEventListener("click", () => {
  if (generating) return;
  history = [];
  messages.replaceChildren();
  addMessage("assistant", "گفت‌وگوی تازه شروع شد.");
  setStatus("مدل آماده است", "WebGPU فعال است؛ تاریخچهٔ مکالمه پاک شد.");
});
stopButton.addEventListener("click", () => stoppingCriteria.interrupt());

composer.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!model || generating || !input.value.trim()) return;
  const text = input.value.trim();
  input.value = "";
  setBusy(true);
  addMessage("user", text);
  const outputNode = addMessage("assistant", "");
  const previousHistory = history.slice();
  const system = "You are Dina, a helpful assistant. Reply in Persian when the user writes Persian. Be accurate and concise.";
  let prompt = "<|im_start|>system\n" + system + "<|im_end|>\n";
  for (const item of previousHistory) {
    if (item.role === "user") {
      prompt += "<|im_start|>user\n" + item.content + "<|im_end|>\n";
      prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    } else {
      prompt += item.content + "<|im_end|>\n";
    }
  }
  prompt += "<|im_start|>user\n" + text + "<|im_end|>\n";
  prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  let generated = "";
  stoppingCriteria.reset();
  try {
    const inputs = await processor(prompt);
    const streamer = new TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (piece) => {
        generated += piece;
        outputNode.textContent = generated;
        messages.scrollTop = messages.scrollHeight;
      },
    });
    await model.generate({
      ...inputs,
      max_new_tokens: 512,
      do_sample: false,
      streamer,
      stopping_criteria: stoppingCriteria,
    });
    history.push({ role: "user", content: text }, { role: "assistant", content: generated.trim() });
    if (!generated.trim()) outputNode.textContent = "تولید متن متوقف شد یا مدل توکنی برنگرداند.";
    setStatus("مدل آماده است", "WebGPU · " + MODEL_ID);
  } catch (error) {
    console.error(error);
    outputNode.textContent = "خطا در تولید پاسخ: " + (error?.message || String(error));
    setStatus("تولید پاسخ ناموفق بود", error?.message || String(error));
  } finally {
    stoppingCriteria.reset();
    setBusy(false);
    input.disabled = false;
    sendButton.disabled = !input.value.trim();
    input.focus();
  }
});

function setBusy(value) {
  generating = value;
  input.disabled = value || !model;
  sendButton.disabled = value || !model || !input.value.trim();
  stopButton.disabled = !value;
  resetButton.disabled = value;
  loadButton.disabled = value || Boolean(model);
}

if (!window.isSecureContext || !navigator.gpu) {
  setStatus("WebGPU هنوز بررسی نشده", "برای این مسیر به WebView سازگار با WebGPU نیاز است. اگر در دسترس نبود، از گفت‌وگوی محلی فعلی استفاده کن.");
}
