# download_model.py
import os
import gdown

url = "https://drive.google.com/uc?export=download&id=1Yy3ItSh6tFbGN75_Vd00kX068GN9jESz"


output = "models/tinyllama-1.1b-chat.Q4_K_M.gguf"

os.makedirs(os.path.dirname(output), exist_ok=True)

print("Downloading TinyLlama model file from Google Drive...")
gdown.download(url, output, quiet=False)
print(f"✅ Model saved to {output}")


