import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai
from groq import Groq
import re
from datetime import datetime

load_dotenv()

# ---- Load model config ----
with open("models.json", "r") as f:
    model_config = json.load(f)

# ---- RateLimiter Class ----
class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.requests = 0
        self.start_time = time.time()

    def wait_if_needed(self):
        current_time = time.time()
        elapsed = current_time - self.start_time

        if elapsed > 60:
            self.requests = 0
            self.start_time = current_time
            print(f"\n[RateLimiter] Starting new minute at {datetime.now().strftime('%H:%M:%S')}")
        elif self.requests >= self.rpm:
            sleep_time = 60 - elapsed
            print(f"\n[RateLimiter] RPM limit hit. Waiting {int(sleep_time)} seconds...")
            time.sleep(sleep_time)
            self.requests = 0
            self.start_time = time.time()
            print(f"[RateLimiter] Resuming at {datetime.now().strftime('%H:%M:%S')}")

        self.requests += 1

# ---- Groq Benchmark ----
class GroqBenchmark:
    def __init__(self, model_name: str, batch_size: int, data_path: str, rate_limiter: RateLimiter):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing GROQ_API_KEY in .env")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_path = data_path
        self.rate_limiter = rate_limiter

    def generate(self, prompt: str) -> str:
        self.rate_limiter.wait_if_needed()
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a mental health diagnosis assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = response.choices[0].message.content.strip()
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return cleaned
        except Exception as e:
            return f"[ERROR]: {e}"

    def run(self, output_path: str):
        df = pd.read_csv(self.data_path)
        if 'query' not in df.columns:
            raise ValueError("'query' column not found in the CSV")

        results = []
        try:
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                queries = batch["query"].tolist()

                prompt = (
                    f"There are exactly {len(queries)} queries. Respond to each with a Yes or No answer, "
                    f"output them separated by only a comma like Yes, No, Yes, etc.:\n\n"
                )
                for idx, q in enumerate(queries, 1):
                    prompt += f"{idx}. {q}\n"

                output = self.generate(prompt)
                print(f"GENERATED OUTPUT (Groq): {output}")
                split_outputs = [x.strip() for x in output.split(",")]
                results.extend(split_outputs)
        finally:
            if len(results) < len(df):
                results += [""] * (len(df) - len(results))
            df[self.model_name] = results[:len(df)]
            df.to_csv(output_path, index=False)
            print(f"[Groq] Saved output to {output_path}")

# ---- Gemini Benchmark ----
class GeminiBenchmark:
    def __init__(self, model_name: str, batch_size: int, data_path: str, rate_limiter: RateLimiter):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.batch_size = batch_size
        self.data_path = data_path
        self.rate_limiter = rate_limiter

    def generate(self, prompt: str) -> str:
        self.rate_limiter.wait_if_needed()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"[ERROR]: {e}"

    def run(self, output_path: str):
        df = pd.read_csv(self.data_path)
        if 'query' not in df.columns:
            raise ValueError("'query' column not found in the CSV")

        results = []
        try:
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                queries = batch["query"].tolist()

                prompt = (
                    f"There are exactly {len(queries)} queries. Respond to each with a Yes or No answer, "
                    f"output them separated by only a comma like Yes, No, Yes, etc.:\n\n"
                )
                for idx, q in enumerate(queries, 1):
                    prompt += f"{idx}. {q}\n"

                output = self.generate(prompt)
                print(f"GENERATED OUTPUT (Gemini): {output}")
                split_outputs = [x.strip() for x in output.split(",")]
                results.extend(split_outputs)
        finally:
            if len(results) < len(df):
                results += [""] * (len(df) - len(results))
            df[self.model_name] = results[:len(df)]
            df.to_csv(output_path, index=False)
            print(f"[Gemini] Saved output to {output_path}")

class Validator:
    def __init__(self, base_model: str, test_model: str, batch_size: int):
        self.test_dir = os.path.join("outputs", f"bs-{batch_size}-" + test_model)
        if not os.path.isdir(self.test_dir):
            raise NotADirectoryError(f"Expected directory not found: {self.test_dir}")

        self.base_model = base_model
        self.test_model = test_model

        self.csv_files = os.listdir(self.test_dir)

    def compare(self, json_output_path):
        results = {}

        for file in self.csv_files:
            path = os.path.join(self.test_dir, file)
            df = pd.read_csv(path)

            if self.base_model not in df.columns or self.test_model not in df.columns:
                print(f"[{file}] Missing expected columns.")
                continue

            base_col = df[self.base_model].astype(str).str.strip().str.lower()
            test_col = df[self.test_model].astype(str).str.strip().str.lower()

            if len(base_col) != len(test_col):
                print(f"[{file}] Row count mismatch.")
                continue

            matches = base_col == test_col
            total = len(matches)
            correct = int(matches.sum())

            yes_count = (test_col == "yes").sum()
            no_count = (test_col == "no").sum()

            accuracy = round((correct / total) * 100, 2)

            results[file] = {
                "match_percentage": accuracy,
                "rows": total,
                "total_matches": correct,
                "yes_count": int(yes_count),
                "no_count": int(no_count)
            }

            print(f"[{file}] Accuracy: {accuracy}% | Yes: {yes_count} | No: {no_count}")

        with open(json_output_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n✅ Saved validation results to {json_output_path}")

# ---- Example Usage ----
# if __name__ == "__main__":
    # data_path = "human_evaluation_binary/DR.csv"
    # batch_size = 2

    # Groq
    # groq_model = "qwen/qwen3-32b"
    # groq_output = "groq_output.csv"
    # groq_benchmark = GroqBenchmark(groq_model, batch_size, data_path)
    # groq_benchmark.run(groq_output)

    # Gemini
    # gemini_model = "gemini-2.0-flash"
    # gemini_output = "gemini_output"
    # gemini_benchmark = GeminiBenchmark(gemini_model, batch_size, data_path)
    # gemini_benchmark.run(gemini_output)