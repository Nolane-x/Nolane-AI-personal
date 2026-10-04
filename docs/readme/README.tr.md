# Nolane AI Personal

**Her promptta yeni bir chatbot gibi sıfırlanmak yerine zaman içinde sürekliliğini korumak için tasarlanmış local-first kişisel AI.**

[English](../../README.md) · **Türkçe**

## v1.0

- **Nolane Presence** — küçük orb soyut bir yüz görevi görür; ışık ve hareket gerçek duruma göre çok hafif değişir.
- **Hafıza ve açık konular** — Nolane'ın neyi hatırladığını ve hangi konuların yarım kaldığını görün; anılar saklanabilir, düzenlenebilir veya unutulabilir.
- **İlişki gelişimi** — “Yeni tanıştık → Tanıdık → Yakın”, XP ya da seviye yok.
- **Üç adımlı ilk kurulum** — hitap şekli, arayüz dili ve konuşma tarzı.
- **Rahatsız etmeyen proaktif konuşma** — Nolane konuşmak istediğinde yalnızca küçük bir kapsül görünür; yok sayılırsa sessiz kalır.
- **Hafif atmosfer** — arka plan mood ile yalnızca çok az değişir.
- Ad ve avatar değişikliği checkpoint veya runtime identity'yi değiştirmez.

## Local-first gizlilik

Durum, hafıza ve süreklilik yerel merkezlidir. Observable Mind yalnızca sınırlandırılmış durum özetleri gösterir; ham gizli muhakeme göstermez. Öğrenme verileri kullanıcı tarafından açıkça incelenmelidir.

## Platformlar

Windows: Tauri + yerel runtime. Android: Tauri + native Rust LocalMobile. Linux: runtime/research araçları. macOS/iOS v1 hedefi değildir.

## Geliştirme

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

Ayrıntılar: [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md), [v1 Living Presence](../V1-LIVING-PRESENCE.md).

v1 yazılım sürümü aynı commit üzerinde CI-verified closure gerektirir ve tüm cihazlar için evrensel pil, termal veya OEM sertifikası iddia etmez.

Apache-2.0.
