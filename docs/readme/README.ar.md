# Nolane AI Personal

**ذكاء اصطناعي شخصي local-first صُمم ليحافظ على الاستمرارية عبر الزمن بدل أن يبدأ من جديد مع كل prompt.**

[English](../../README.md) · **العربية**

## v1.0

- **Nolane Presence** — كرة صغيرة تعمل كوجه تجريدي وتعكس الحالة الحقيقية بتغييرات ضوئية وحركية خفيفة.
- **الذكريات والمواضيع المفتوحة** — شاهد ما يتذكره Nolane وما بقي غير مكتمل؛ يمكن الاحتفاظ بالذاكرة أو تعديلها أو نسيانها.
- **نمو العلاقة** — «تعارف جديد → مألوف → قريب» من دون XP أو مستويات.
- **إعداد أولي من ثلاث خطوات** — الاسم المفضل ولغة الواجهة وأسلوب الحديث.
- **محادثة استباقية بلا إزعاج** — عندما يريد Nolane الكلام تظهر كبسولة صغيرة فقط؛ تجاهلها فيبقى هادئًا.
- **أجواء خفيفة** — تتغير الخلفية بدرجة بسيطة جدًا مع mood.
- تغيير الاسم أو الصورة لا يغير checkpoint أو هوية runtime الداخلية.

## الخصوصية local-first

الحالة والذاكرة والاستمرارية تتمحور حول التخزين المحلي. Observable Mind يعرض ملخصات محدودة للحالة ولا يعرض raw hidden reasoning. بيانات التعلم تحتاج مراجعة صريحة من المستخدم.

## المنصات

Windows: ‏Tauri + runtime محلي. Android: ‏Tauri + LocalMobile أصلي بلغة Rust. Linux: أدوات runtime/research. macOS/iOS ليست ضمن أهداف v1.

## التطوير

```bash
python -m pip install -e .
nolane-personal init
nolane-personal run --no-model --tick-seconds 5
```

```bash
cd apps/product-client
npm install
npm run tauri -- dev
```

راجع [Architecture](../ARCHITECTURE.md) و[Roadmap](../ROADMAP.md) و[v1 Living Presence](../V1-LIVING-PRESENCE.md).

يتطلب إصدار v1 البرمجي CI-verified closure للـ commit نفسه، ولا يدّعي اعتمادًا شاملًا للبطارية أو الحرارة أو OEM.

Apache-2.0.
