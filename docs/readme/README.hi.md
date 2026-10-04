# Nolane AI Personal

**एक local-first व्यक्तिगत AI जो हर prompt पर नया chatbot बनने के बजाय समय के साथ अपनी निरंतरता बनाए रखने के लिए बनाया गया है।**

[English](../../README.md) · **हिन्दी**

## v1.0

- **Nolane Presence** — छोटा orb एक अमूर्त चेहरे की तरह काम करता है और वास्तविक state के अनुसार हल्की रोशनी/गतिशीलता बदलता है।
- **यादें और अधूरी बातें** — देखें कि Nolane क्या याद रखता है और कौन-सी बातें बाकी हैं; memory को रखना, संपादित करना या भूलना संभव है।
- **रिश्ते का विकास** — “नई पहचान → परिचित → करीबी”, बिना XP या level के।
- **तीन-चरण onboarding** — पसंदीदा नाम, UI भाषा और बातचीत का ढंग।
- **बिना बाधा proactive conversation** — Nolane कुछ कहना चाहे तो केवल एक छोटा capsule दिखता है; अनदेखा करने पर वह शांत रहता है।
- **हल्का conversation atmosphere** — mood के साथ background बहुत थोड़ा बदलता है।
- नाम/avatar बदलना checkpoint या runtime identity को नहीं बदलता।

## Local-first गोपनीयता

State, memory और continuity स्थानीय हैं। Observable Mind सीमित state summaries दिखाता है, raw hidden reasoning नहीं। Learning data के लिए उपयोगकर्ता की स्पष्ट समीक्षा आवश्यक है।

## प्लेटफ़ॉर्म

Windows: Tauri + local runtime. Android: Tauri + native Rust LocalMobile. Linux: runtime/research tooling. macOS/iOS v1 target नहीं हैं।

## Development

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

विवरण: [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md), [v1 Living Presence](../V1-LIVING-PRESENCE.md).

v1 software release के लिए उसी commit पर CI-verified closure आवश्यक है; यह सभी devices के battery, thermal या OEM certification का दावा नहीं करता।

Apache-2.0.
