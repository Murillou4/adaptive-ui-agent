# Guia Completo - Adaptive UI Agent (Universal)

## Visão Geral
Este documento detalha a implementação do Agente Universal de UI Adaptativa ("Adaptive UI Agent"), capaz de interagir com qualquer aplicativo ou jogo através de visão computacional e controle de cursor/teclado.

## Arquitetura do Sistema

O sistema é composto por 5 subsistemas principais:

### 1. Sistema de Visão Universal (`vision/`)
Responsável por "ver" a tela e identificar elementos.
-   **YOLO-World (`yolo_backend.py`)**: Detecção de objetos com vocabulário aberto. Detecta "botões", "ícones", "personagens", etc. sem treino específico.
-   **OCR (`ocr_backend.py`)**: Extração de texto usando EasyOCR. Suporta múltiplos idiomas.
-   **Universal Detector (`universal_detector.py`)**: API unificada que roteia consultas para o backend apropriado (texto ou objeto).
-   **State Extractor (`state_extractor.py`)**: Converte detecções brutas em um estado estruturado `VisualState`.

### 2. Ambiente Universal (`env/`)
A interface entre o agente e o sistema operacional.
-   **Screen Capture (`interaction/screen_capture.py`)**: Captura de tela de alta performance (60fps+) usando `mss`. Suporta captura de janelas específicas.
-   **Input Controller (`interaction/input_controller.py`)**: Simulação de Mouse/Teclado via `pynput` e `pyautogui`.
-   **AppSandbox (`env/app_sandbox.py`)**: Wrapper de segurança que restringe a visão e ações do agente a uma janela específica.
-   **UniversalEnv (`env/universal_env.py`)**: Interface padrão Gymnasium (RL) para qualquer app.

### 3. Biblioteca de Habilidades (`skills/`)
Comportamentos reutilizáveis (Skills) que o agente pode executar.
-   **Cursor Skills**: `MoveTo`, `Click`, `Drag`.
-   **Keyboard Skills**: `TypeText`, `Hotkey`.
-   **Navigation Skills**: `Scroll`, `SwitchWindow`.
-   **Persistence**: Skills podem ser salvas/carregadas e versionadas.

### 4. Controlador Hierárquico (`controller/`)
O "cérebro" executivo do agente.
-   **MetaController (`controller/meta_controller.py`)**: Planeja sequências de skills para atingir objetivos de alto nível (e.g., "Pesquisar gatos no Google").
-   **Recuperação de Falhas**: Se uma skill falha (ex: botão não encontrado), o controlador tenta estratégias de recuperação (esperar, pressionar ESC) antes de abortar.

### 5. Infraestrutura de Treino (`training/`)
-   **Curriculum Manager (`training/curriculum_manager.py`)**: Gerencia o progresso do aprendizado baseado em arquivos YAML.
-   **Auto Curriculum**: Gera novas tarefas automaticamente via LLM.

## Como Usar

### Instalação

```bash
pip install -r requirements.txt
```

### Executando o Agente (Demo)

```bash
# Exemplo: Pesquisar no Google (Requer navegador aberto)
python run_agent.py --goal "Search for cats on Google"
```

### Executando Testes

```bash
python -m pytest tests/ -v
```

## Próximos Passos (Roadmap)

1.  **Integração Real com LLM**: Atualmente o MetaController usa uma heurística para demos. Conectar o `LLMPlanner` real para gerar planos complexos.
2.  **Treinamento em Larga Escala**: Usar o curriculum manager para treinar o agente em 1000+ tarefas.
3.  **VMM/Docker**: Implementar sandboxing mais robusto.
