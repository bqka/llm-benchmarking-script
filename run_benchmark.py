from script import GeminiBenchmark
import os

batch_size = 5
data_path = "human_evaluation_binary"
gemini_model = "models/gemini-2.0-flash"
output_dir = os.path.join("outputs", gemini_model)
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(data_path):
    print(f"Running on file {file}...")
    file_path = os.path.join(data_path, file)
    gemini_output = os.path.join(output_dir, file)

    gemini_benchmark = GeminiBenchmark(gemini_model, batch_size, file_path)
    gemini_benchmark.run(gemini_output)