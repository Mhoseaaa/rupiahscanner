import torch
import torchvision
import torchaudio

print("Torch version:", torch.__version__) # Torch version: 2.6.0+cu126
print("Torchvision version:", torchvision.__version__) # Torchvision version: 0.21.0+cu126
print("Torchaudio version:", torchaudio.__version__) # Torchaudio version: 2.6.0+cu126
print("CUDA Available:", torch.cuda.is_available()) # CUDA Available: True
print("CUDA Version:", torch.version.cuda) # CUDA Version: 12.6
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")