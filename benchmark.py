from utils import GeminiBenchmark, RateLimiter
import os
import json

BATCH_SIZE = 5
DATA_PATH = "human_evaluation_binary"

gemini_model = "models/gemini-2.0-flash"
output_dir = os.path.join("outputs", gemini_model)

### RATE LIMITER
with open("models.json", "r") as f:
    model_config = json.load(f)
rpm = next((m["rpm"] for m in model_config["gemini-models"] if m["name"] == gemini_model), 60)

rate_limiter = RateLimiter(rpm)
###

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(DATA_PATH):
    print(f"Running on file {file}...")
    file_path = os.path.join(DATA_PATH, file)
    gemini_output = os.path.join(output_dir, file)

    gemini_benchmark = GeminiBenchmark(gemini_model, BATCH_SIZE, file_path, rate_limiter)
    gemini_benchmark.run(gemini_output)