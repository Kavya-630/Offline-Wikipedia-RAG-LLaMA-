# download_model.py
import os
import gdown

url = "https://drive.google.com/uc?export=download&id=14skIKTtf74W8aB95lyLgYR58M8CiQScm"


output = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

os.makedirs(os.path.dirname(output), exist_ok=True)

print("Downloading Llama model file from Google Drive...")
gdown.download(url, output, quiet=False)
print(f"✅ Model saved to {output}")




