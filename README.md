# Adaptive UI Agent

> **Agente de RL Visual que aprende a executar tarefas UI a partir de linguagem natural**

Baseado no paper **arXiv 2312.01203v3**: "Harnessing Discrete Representations for Continual RL"

---

## 🧠 Arquitetura

```
User (texto) → LLM → Plan (JSON) → Translator → Reward → RL (PPO) → Ações
                                                            ↑
                                    Pixels → VQ-VAE → Multi-One-Hot
```

> **LLM nunca toca no mouse. RL nunca lê texto.**

---

## ⚡ Quick Start

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Configurar LLM (copie e edite)
cp configs/llm_config.example.yaml configs/llm_config.yaml
# Edite com sua API key

# 3. Executar modo interativo
python planner/integration.py --interactive
```

---

## 🔑 Providers LLM Suportados

| Provider | Modelos | API Key Env |
|----------|---------|-------------|
| OpenAI | gpt-5.2-instant/thinking/pro | `OPENAI_API_KEY` |
| Anthropic | claude-4.5-opus/sonnet/haiku | `ANTHROPIC_API_KEY` |
| Google | gemini-2.5-flash/pro | `GOOGLE_API_KEY` |
| xAI | grok-4, grok-4.1 | `XAI_API_KEY` |
| Ollama | llama4, qwen3, gemma3 | (local) |

```bash
# Uso com provider específico
python planner/integration.py --provider anthropic --interactive
```

---

## 📖 Documentação

- **[Guia Completo](docs/GUIA_COMPLETO.md)** - Instalação, configuração, uso detalhado
- **[LLM Config](configs/llm_config.example.yaml)** - Configuração de API keys
- **[Arquitetura](docs/GUIA_COMPLETO.md#arquitetura)** - Diagrama e fluxo de dados

---

## 📁 Estrutura

```
├── planner/          # LLM-RL Integration
│   ├── goal_dsl.py       # Visual vocabulary
│   ├── llm_planner.py    # LLM → Plan
│   ├── llm_provider.py   # Universal LLM (LiteLLM)
│   └── integration.py    # Full pipeline
├── vision/           # VQ-VAE (discrete encoding)
├── agent/            # PPO (policy learning)
├── env/              # Pygame sandbox (64×64)
└── configs/          # Hyperparameters + LLM keys
```

---

## 🎯 Exemplo

```python
from planner import LLMRLIntegration

agent = LLMRLIntegration(llm_provider="openai")
result = agent.train_on_goal("Cria 3 quadrados azuis alinhados")

print(f"Sucesso: {result.success}")
print(f"Taxa: {result.final_success_rate:.1%}")
```

---

*Janeiro 2026 | Paper: arXiv 2312.01203v3*
