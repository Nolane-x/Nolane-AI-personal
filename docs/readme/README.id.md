# Nolane AI Personal

**AI personal local-first yang dirancang untuk mempertahankan kesinambungan sepanjang waktu, bukan memulai ulang seperti chatbot baru pada setiap prompt.**

[English](../../README.md) · **Bahasa Indonesia**

## v1.0

- **Nolane Presence** — orb kecil menjadi wajah abstrak; cahaya dan geraknya berasal dari state runtime yang nyata.
- **Memori & hal yang belum selesai** — lihat apa yang diingat Nolane dan topik yang masih terbuka; memori dapat disimpan, diedit, atau dilupakan.
- **Pertumbuhan hubungan** — “Baru kenal → Akrab → Dekat”, tanpa XP atau level.
- **Onboarding tiga langkah** — nama panggilan, bahasa UI, dan gaya bicara.
- **Percakapan proaktif tanpa mengganggu** — saat Nolane ingin bicara, hanya muncul kapsul kecil; jika diabaikan Nolane tetap diam.
- **Atmosfer halus** — latar berubah sangat sedikit mengikuti mood.
- Nama/avatar dapat diubah tanpa mengganti checkpoint atau runtime identity.

## Privasi local-first

State, memori, dan kontinuitas berpusat di perangkat lokal. Observable Mind hanya menampilkan ringkasan state yang dibatasi, bukan raw hidden reasoning. Data pembelajaran memerlukan review eksplisit dari pengguna.

## Platform

Windows: Tauri + runtime lokal. Android: Tauri + LocalMobile native Rust. Linux: runtime/research tooling. macOS/iOS bukan target v1.

## Pengembangan

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

Lihat [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md), dan [v1 Living Presence](../V1-LIVING-PRESENCE.md).

Rilis software v1 memerlukan CI-verified closure pada commit yang sama dan tidak mengklaim sertifikasi universal baterai, termal, atau OEM.

Apache-2.0.
