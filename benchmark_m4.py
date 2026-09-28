import torch
import time

# Taille de la matrice (ajuste pour faire chauffer le M4)
size = 8000 

def benchmark(device_name):
    device = torch.device(device_name)
    # Création de deux matrices géantes
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm-up (pour réveiller le GPU)
    _ = torch.matmul(a, b)
    
    start = time.time()
    # Multiplication de matrice (opération de base du Deep Learning)
    result = torch.matmul(a, b)
    
    # Attendre que le GPU finisse (important pour le benchmark)
    if device_name == "mps":
        torch.mps.synchronize()
        
    end = time.time()
    return end - start

print(f"⏳ Benchmark sur matrice {size}x{size}...")
cpu_time = benchmark("cpu")
print(f"🔴 Temps CPU : {cpu_time:.4f} secondes")

if torch.backends.mps.is_available():
    gpu_time = benchmark("mps")
    print(f"🔵 Temps GPU (M4 MPS) : {gpu_time:.4f} secondes")
    print(f"🚀 Gain : {cpu_time/gpu_time:.1f}x plus rapide")
