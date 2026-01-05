# Adaptive UI Agent - Guia Completo de Uso

> **Agente de RL Visual com Integração LLM**  
> Aprende a executar tarefas visuais a partir de objetivos em linguagem natural.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Instalação](#instalação)
3. [Configuração de LLM](#configuração-de-llm)
4. [Uso Básico](#uso-básico)
5. [Arquitetura](#arquitetura)
6. [Comandos do Chat](#comandos-do-chat)
7. [Treinamento](#treinamento)
8. [Exemplos Práticos](#exemplos-práticos)
9. [FAQ](#faq)

---

## 🧠 Visão Geral

O Adaptive UI Agent é um sistema que combina:

| Componente | Função |
|------------|--------|
| **LLM** | Interpreta goals → gera planos estruturados |
| **VQ-VAE** | Comprime pixels → representação discreta |
| **PPO** | Aprende política de ações via RL |
| **Visual Detectors** | Detecta elementos por pixels |

### Princípio Fundamental

```
LLM = cérebro simbólico (intenção)
RL = cérebro motor (ação)
VQ-VAE = ponte visual (percepção)
```

> ⚠️ **LLM nunca toca no mouse. RL nunca lê texto.**

---

## 🚀 Instalação

### Requisitos
- Python 3.10+
- CUDA (opcional, para GPU)

### Passos

```bash
# 1. Clone ou acesse o diretório
cd adaptive-ui-agent

# 2. Instale dependências
pip install -r requirements.txt

# 3. Instale LiteLLM para suporte a múltiplos LLMs
pip install litellm

# 4. (Opcional) Instale dependências de desenvolvimento
pip install -e ".[dev]"
```

### Verificação

```bash
# Testar ambiente
python -c "from env import SandboxEnv; print('Env OK')"

# Testar VQ-VAE
python -c "from vision import VQVAE; print('VQ-VAE OK')"

# Testar Planner
python -c "from planner import create_planner; print('Planner OK')"
```

---

## 🔑 Configuração de LLM

### Passo 1: Copie o arquivo de exemplo

```bash
cp configs/llm_config.example.yaml configs/llm_config.yaml
```

### Passo 2: Configure sua API Key

Edite `configs/llm_config.yaml`:

```yaml
# Escolha seu provider
default_provider: "openai"  # openai, anthropic, google, xai, ollama

# Configure a API key
openai:
  api_key: "sk-sua-chave-aqui"  # Ou use ${OPENAI_API_KEY}
  model: "gpt-5.2-instant"      # Modelo a usar
```

### Providers Suportados

| Provider | Modelos | Notas |
|----------|---------|-------|
| **OpenAI** | gpt-5.2-instant/thinking/pro | Mais evoluído |
| **Anthropic** | claude-4.5-opus/sonnet/haiku | Contexto longo |
| **Google** | gemini-2.5-flash/pro, gemini-3 | Multimodal |
| **xAI** | grok-4, grok-4.1 | Tempo real |
| **Ollama** | llama4, qwen3, gemma3 | Local/grátis |
| **LiteLLM** | 100+ modelos | Universal |

### Via Variáveis de Ambiente

```bash
# Alternativa: configure via env vars
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

---

## 💻 Uso Básico

### 1. Modo Interativo (Recomendado)

```bash
python planner/integration.py --interactive
```

```
🎯 Goal: Cria 3 quadrados azuis alinhados

✅ Plan generated:
   Goal: create_three_squares
   Elements: 1
   Constraints: 1

🎯 Starting training...
Episode 50: success_rate=23%, progress=45%
Episode 100: success_rate=67%, progress=82%
Episode 150: success_rate=91%, progress=98%

🎉 Goal achieved! Success rate: 91%
```

### 2. Goal Único

```bash
python planner/integration.py --goal "Cria um botão centralizado"
```

### 3. Com Provider Específico

```bash
# Usar Claude
python planner/integration.py --provider anthropic --interactive

# Usar modelo local
python planner/integration.py --provider ollama --interactive
```

### 4. Script Python

```python
from planner import LLMRLIntegration

# Inicializar
integration = LLMRLIntegration(
    config_path="configs/default.yaml",
    llm_provider="openai"
)

# Treinar em um objetivo
result = integration.train_on_goal(
    "Cria 3 quadrados azuis alinhados",
    max_episodes=500,
    success_threshold=0.8
)

print(f"Sucesso: {result.success}")
print(f"Taxa de sucesso: {result.final_success_rate:.1%}")
```

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    ADAPTIVE UI AGENT                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │    User     │───▶│ LLM Planner │───▶│Structured   │     │
│  │   (texto)   │    │  (GPT/etc)  │    │   Plan      │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                               │             │
│                                               ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Actions   │◀───│  PPO Agent  │◀───│ Objective   │     │
│  │  (mouse)    │    │  (policy)   │    │ Translator  │     │
│  └──────┬──────┘    └──────▲──────┘    └─────────────┘     │
│         │                  │                                │
│         ▼                  │                                │
│  ┌─────────────┐    ┌──────┴──────┐    ┌─────────────┐     │
│  │ Environment │───▶│   VQ-VAE    │───▶│Multi-One-Hot│     │
│  │  (pygame)   │    │  (encoder)  │    │  (18,432-d) │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Fluxo de Dados

1. **User** → texto em linguagem natural
2. **LLM** → decompõe em plan JSON (usando DSL)
3. **Translator** → converte plan em reward function
4. **Environment** → pixels 64×64 RGB
5. **VQ-VAE** → 6×6 latent → 18,432-dim multi-one-hot
6. **PPO** → aprende política de ações
7. **Actions** → 9 ações (8 direções + click)

---

## 🎮 Comandos do Chat

Durante o modo interativo:

| Comando | Descrição |
|---------|-----------|
| `status` | Mostra estado atual do agente |
| `set_rule <rule>` | Muda regra (blue_bad, blue_good) |
| `swap_targets` | Inverte target/obstacle (Continual RL) |
| `screenshot` | Captura tela atual |
| `reconstruct` | Mostra reconstrução VQ-VAE |
| `show_latent` | Visualiza multi-one-hot |
| `step <action>` | Executa ação (0-8) |
| `reset` | Reseta ambiente |
| `pause` / `resume` | Controle de treino |
| `help` | Lista comandos |
| `quit` | Sair |

---

## 🎯 Treinamento

### Pipeline Completo

```bash
# Executa: dataset → VQ-VAE → PPO
python scripts/run_training.py
```

### Etapas Individuais

```bash
# 1. Gerar dataset (5000 screenshots)
python scripts/run_training.py --skip-vqvae --skip-ppo

# 2. Treinar VQ-VAE
python scripts/run_training.py --skip-ppo

# 3. Treinar PPO
python scripts/run_training.py --skip-dataset --skip-vqvae
```

### Monitoramento (TensorBoard)

```bash
tensorboard --logdir runs/
```

---

## 📝 Exemplos Práticos

### Exemplo 1: Layout Simples

```python
from planner import LLMRLIntegration

agent = LLMRLIntegration()
result = agent.train_on_goal("3 retângulos azuis em linha")
# Agente aprende a criar e alinhar elementos
```

### Exemplo 2: Continual RL

```python
# Treina primeira regra
agent.train_on_goal("Clica no quadrado azul")

# Muda regra (continual RL)
agent.env.swap_targets()

# Agente re-adapta rapidamente
agent.train_on_goal("Clica no quadrado vermelho", max_episodes=200)
```

### Exemplo 3: Com LLM Real

```python
from planner import LLMRLIntegration
from planner.llm_provider import create_llm_provider

# Usar GPT-5.2
provider = create_llm_provider(provider="openai", model="gpt-5.2-thinking")

agent = LLMRLIntegration(llm_provider=provider)
result = agent.train_on_goal("Cria uma interface de login simples")
```

---

## ❓ FAQ

### Q: O agente "entende" o que está fazendo?
**A:** Não. Ele aprende associações visuais → ações → recompensas. Não há compreensão semântica.

### Q: Posso usar sem GPU?
**A:** Sim. CPU funciona, mas treinamento é mais lento.

### Q: Como adicionar novo provider LLM?
**A:** Edite `configs/llm_config.yaml` e use o formato LiteLLM: `provider/model`.

### Q: O agente pode criar interfaces reais?
**A:** Em teoria, sim (com Figma em VM). Na prática, começa com sandbox simples.

### Q: Qual a diferença para RPA tradicional?
**A:** RPA usa scripts fixos. Este agente **aprende** políticas a partir de pixels.

---

## 📁 Estrutura do Projeto

```
adaptive-ui-agent/
├── agent/              # PPO implementation
│   ├── ppo.py          # Policy/Value networks
│   └── trainer.py      # Training orchestrator
├── configs/
│   ├── default.yaml    # Hyperparameters
│   └── llm_config.yaml # LLM API keys
├── env/
│   ├── sandbox_env.py  # Pygame environment
│   └── extended_sandbox.py
├── interaction/
│   ├── chat_interface.py
│   └── visualizer.py
├── planner/            # LLM-RL integration
│   ├── goal_dsl.py     # Visual vocabulary
│   ├── visual_detectors.py
│   ├── reward_generator.py
│   ├── llm_planner.py
│   ├── llm_provider.py # Universal LLM (LiteLLM)
│   └── integration.py
├── scripts/
│   ├── run_training.py
│   └── demo.py
├── tests/
│   ├── test_env.py
│   ├── test_vqvae.py
│   └── test_ppo.py
└── vision/
    ├── vqvae.py
    └── train_vqvae.py
```

---

## 🔗 Links Úteis

- **Paper base**: arXiv 2312.01203v3
- **LiteLLM docs**: https://docs.litellm.ai
- **TensorBoard**: localhost:6006 (após `tensorboard --logdir runs/`)

---

*Última atualização: Janeiro 2026*
