# 🔍 LLM Benchmarking Script

This repository benchmarks large language models (LLMs) by generating results in a CSV file. Currently, it supports **binary classification tasks** with **Yes** or **No** answers.

---

## 🚀 Features

- Benchmark LLMs using batch prompts
- Supports:
  - Google Gemini models (via Vertex AI or Generative AI SDK)
  - Groq-hosted models (e.g., LLaMA, Mixtral, Qwen)
- Automatic rate limiting based on model-specific RPM (requests per minute)
- Saves partial results even if interrupted
- Cleans unnecessary `<think>` blocks from Groq model responses
- Saves outputs in structured CSV format

---

## 📁 Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/llm-benchmarking-script.git
cd llm-benchmarking-script
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add API Keys

Create a .env file in the root directory:

```bash
GEMINI_API_KEY=your_google_gemini_key
GROQ_API_KEY=your_groq_api_key
```

## Input Format

Input should be a CSV file (or multiple files) with a column named:

query

Each row in this column should contain a question to be answered with "Yes" or "No".

Place input files under the human_evaluation_binary/ directory.

## Output Format

- Model outputs are saved in the outputs/{model_name}/ folder.

- Each file is saved with the same name as the input CSV, but with an added column for the model responses.

- Accuracy results are saved under results/{model_name}