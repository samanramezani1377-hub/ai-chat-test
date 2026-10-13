# بستهٔ مدل محلی MiniLM برای فرمان‌های دینا

تمام inference درون برنامه و روی دستگاه اجرا می‌شود؛ هیچ API یا سرویس ابری لازم نیست.

## ساختار ZIP

بسته باید دقیقاً این فایل‌های اصلی را داشته باشد (وجود پوشهٔ والد داخل ZIP اشکالی ندارد):

- `model.onnx`: **مدل fine-tuned برای sequence classification** با خروجی logits به شکل `[1, number_of_labels]`.
- `tokenizer.json`: فایل tokenizer سازگار Hugging Face، با مدل WordPiece یا BPE سبک SentencePiece/XLM-R.
- `labels.json`: ترتیب برچسب‌ها دقیقاً مطابق ترتیب logitهای مدل.

نمونهٔ `labels.json`:

```json
{
  "labels": [
    "CHAT",
    "GET_TIME",
    "GET_DATE",
    "INCREASE_VOLUME",
    "DECREASE_VOLUME",
    "SET_VOLUME",
    "INCREASE_BRIGHTNESS",
    "DECREASE_BRIGHTNESS",
    "SET_BRIGHTNESS"
  ]
}
```

می‌توان به‌جای شیء دارای `labels`، آرایهٔ مستقیم رشته‌ها را نیز استفاده کرد. برچسب‌ها باید با ترتیب خروجی مدل یکسان باشند.

## قرارداد ورودی ONNX

- `input_ids`: اعداد صحیح ۶۴ بیتی، شکل `[1, 64]`
- `attention_mask`: اعداد صحیح ۶۴ بیتی، شکل `[1, 64]`
- `token_type_ids`: اختیاری؛ اگر مدل آن را درخواست کند، صفرهای ۶۴تایی ارسال می‌شوند.
- خروجی اول: tensor از نوع FLOAT با یک logit برای هر برچسب.

توکن‌های شروع، پایان، ناشناخته و padding باید در واژگان tokenizer موجود باشند. مدل باید با همان tokenizer و برچسب‌های مورد استفاده در زمان fine-tuning صادر شده باشد.

## نکتهٔ مهم دربارهٔ MiniLM

مدل عمومی sentence-embedding به‌تنهایی intent classifier نیست و نباید مستقیماً به‌عنوان طبقه‌بند فرمان وارد شود. ابتدا باید با نمونه‌های فارسی برای intentهای بالا fine-tune و به ONNX با سرِ classification صادر شود. طبقه‌بند داخل برنامه فقط پیش‌بینی‌هایی را می‌پذیرد که confidence و فاصلهٔ آن‌ها از گزینهٔ دوم از آستانهٔ محافظه‌کارانه عبور کند. اگر بسته ناسازگار باشد، خروجی کم‌اطمینان باشد یا مدل وارد نشده باشد، فرمان‌های محلی به قواعد محدود و محافظه‌کارانه برمی‌گردند؛ متن‌های عادی همچنان به مسیر گفت‌وگو می‌روند.

## فرمان‌های دستگاه

- `GET_TIME`، `GET_DATE`
- `INCREASE_VOLUME`، `DECREASE_VOLUME`، `SET_VOLUME`
- `INCREASE_BRIGHTNESS`، `DECREASE_BRIGHTNESS`، `SET_BRIGHTNESS`

برای `SET_VOLUME` و `SET_BRIGHTNESS` مدل باید عدد هدف را از متن واقعی فرمان استخراج کند؛ کد عدد را از خروجی مدل حدس نمی‌زند. اگر کاربر درصد صدا را صریح بگوید، مقدار به‌عنوان درصد اعمال می‌شود؛ عدد بدون درصد برای صدا می‌تواند سطح مستقیم دستگاه باشد.
