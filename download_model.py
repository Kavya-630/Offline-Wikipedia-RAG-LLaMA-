# download_model.py
import os
import gdown

url = "https://drive.google.com/uc?id=1mgbDYKRpG0F4hMpvAZ2uUgnHuTFV21zL"

output = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

os.makedirs(os.path.dirname(output), exist_ok=True)

print("Downloading model file from Google Drive...")
gdown.download(url, output, quiet=False)
print(f"✅ Model saved to {output}")
