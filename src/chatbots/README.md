# Chatbots Streamlit Application
Streamlit을 사용하여 구축된 다양한 챗봇 기능을 제공하는 어플리케이션입니다.

## 실행 방법
```sh
streamlit run src/chatbots/main.py
```

## 로컬 LLM 설정 (Ollama & Llama 3.2)
이 섹션은 로컬 환경에서 Ollama를 사용하여 Llama 3.2와 같은 대규모 언어 모델(LLM)을 직접 실행하는 방법을 나타냅니다.

### 1. Ollama 설치
*   **macOS:** [ollama.com](https://ollama.com)에서 다운로드하여 설치합니다.
*   **Windows:** [ollama.com](https://ollama.com)에서 다운로드하여 설치합니다.

### 2. Llama 3.2 모델 실행
Ollama 설치가 완료되면 다음 단계를 따라 Llama 3.2 모델을 실행할 수 있습니다.

#### a. Ollama 서버 실행 (필요한 경우)
일반적으로 Ollama를 설치하면 백그라운드에서 자동으로 실행됩니다. 만약 데스크톱 애플리케이션 없이 Ollama를 시작하고 싶다면 다음 명령어를 사용합니다:
```bash
ollama serve
```
이 명령어는 Ollama 서버를 시작합니다. 별도의 터미널 창을 열어 다음 단계를 진행하세요.

#### b. Llama 3.2 모델 다운로드 및 실행
다음 명령어를 사용하여 Llama 3.2 모델을 다운로드하고 대화형으로 실행할 수 있습니다:
```bash
ollama run llama3.2
```
이 명령어를 처음 실행하면 `llama3.2` 모델을 다운로드합니다. 다운로드가 완료되면 바로 모델과 대화를 시작할 수 있습니다.

**참고:** `llama3.2`는 여러 크기의 모델이 있을 수 있습니다 (예: 1B, 3B). 특정 크기를 지정하려면 다음과 같이 태그를 사용할 수 있습니다:
```bash
ollama run llama3.2:3b # 3B 모델 실행
ollama run llama3.2:1b # 1B 모델 실행
```
가장 일반적인 `llama3.2`는 보통 중간 크기의 모델(예: 3B)을 가리킵니다.
