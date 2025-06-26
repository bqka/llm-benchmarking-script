import pandas as pd
import os

folder_path = "human_evaluation/test_instruction_expert"
output_folder = "human_evaluation_binary"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(folder_path):
    
    if not file.endswith(".csv"):
        print("Not a csv file:", file)
        continue

    print(f"Opening {file}...")

    df = pd.read_csv(os.path.join(folder_path, file))

    ans = []
    is_binary = True

    for response in df["gpt-3.5-turbo"]:
        if(response.strip().lower().startswith("n")):
            ans.append("No")
        elif(response.strip().lower().startswith("y")):
            ans.append("Yes")
        else:
            is_binary = False
            break
    if not is_binary:
        print(f"{file} is not binary. Skipping...")
        continue
    df["gpt-3.5-turbo"] = ans
    df.to_csv(os.path.join(output_folder, file))

print("Done")