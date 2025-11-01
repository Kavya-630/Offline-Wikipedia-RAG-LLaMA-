# download_model.py
import os
import gdown

url = "https://drive.google.com/uc?export=download&id=1bquBi_ccK4XDsatiHZsucysPUBXzmga6"


output = "models/phi-3-mini-4k-instruct.Q4_K_M.gguf"

os.makedirs(os.path.dirname(output), exist_ok=True)

print("Downloading Llama model file from Google Drive...")
gdown.download(url, output, quiet=False)
print(f"✅ Model saved to {output}")






