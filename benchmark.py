from utils import *
import os
import json

BATCH_SIZE = 20

BASE_MODEL = "gpt-3.5-turbo"
TEST_MODEL = "gemini-2.0-flash-lite"

DATA_PATH = os.path.join("data", BASE_MODEL)
OUTPUT_DIR = os.path.join("outputs", f"bs-{BATCH_SIZE}-" + TEST_MODEL)
RESULT_PATH = os.path.join("results", f"bs-{BATCH_SIZE}-" + TEST_MODEL + ".json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

### Load model config and detect model type
with open("models.json", "r") as f:
    model_config = json.load(f)

if any(m["name"] == TEST_MODEL for m in model_config.get("gemini-models", [])):
    model_type = "gemini"
    rpm = next(m["rpm"] for m in model_config["gemini-models"] if m["name"] == TEST_MODEL)
    BenchmarkClass = GeminiBenchmark

elif any(m["name"] == TEST_MODEL for m in model_config.get("groq-models", [])):
    model_type = "groq"
    rpm = next(m["rpm"] for m in model_config["groq-models"] if m["name"] == TEST_MODEL)
    BenchmarkClass = GroqBenchmark

else:
    raise ValueError(f"Model '{TEST_MODEL}' not found in models.json")

rate_limiter = RateLimiter(rpm)

### Run benchmark
for file in os.listdir(DATA_PATH):
    if not file.endswith(".csv"):
        continue
    print(f"Running {TEST_MODEL} on file: {file}...")

    file_path = os.path.join(DATA_PATH, file)
    output_path = os.path.join(OUTPUT_DIR, file)

    benchmark = BenchmarkClass(TEST_MODEL, BATCH_SIZE, file_path, rate_limiter)
    benchmark.run(output_path)

## Validate Model
validator = Validator(BASE_MODEL, TEST_MODEL, BATCH_SIZE)
validator.compare(RESULT_PATH)