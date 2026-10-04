# Nolane AI Personal

**Uma IA pessoal local-first feita para manter continuidade ao longo do tempo, em vez de reiniciar como um chatbot novo a cada prompt.**

[English](../../README.md) · **Português**

## v1.0

- **Nolane Presence** — um pequeno orbe funciona como rosto abstrato e reflete o estado real com luz e movimento discretos.
- **Memória e assuntos pendentes** — veja o que Nolane lembra e o que ficou em aberto; memórias podem ser mantidas, editadas ou esquecidas.
- **Crescimento da relação** — “Recém-conhecidos → Familiar → Próximos”, sem XP ou níveis.
- **Onboarding em três passos** — nome preferido, idioma da interface e estilo de conversa.
- **Conversa proativa sem interrupção** — quando Nolane quer falar, aparece apenas uma pequena cápsula; se ignorada, ele permanece quieto.
- **Atmosfera sutil** — o fundo muda apenas alguns por cento conforme o mood.
- Nome e avatar podem mudar sem alterar checkpoint nem identidade interna do runtime.

## Privacidade local-first

Estado, memória e continuidade são locais. Observable Mind mostra apenas resumos limitados de estado, nunca raciocínio privado bruto. Dados de aprendizagem exigem revisão explícita do usuário.

## Plataformas

Windows: Tauri + runtime local. Android: Tauri + LocalMobile nativo em Rust. Linux: ferramentas de runtime/pesquisa. macOS/iOS não são alvos da v1.

## Desenvolvimento

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

Veja [Architecture](../ARCHITECTURE.md), [Roadmap](../ROADMAP.md) e [v1 Living Presence](../V1-LIVING-PRESENCE.md).

O release de software v1 exige closure CI verificada no mesmo commit e não reivindica certificação universal de bateria, temperatura ou OEM.

Apache-2.0.
