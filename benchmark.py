# benchmark.py
import time
import numpy as np
import torch
import onnxruntime as ort
from train import QuadraticModel # Importa a arquitetura do modelo

# --- PARÂMETROS DO BENCHMARK ---
NUM_RUNS = 500  # Número de inferências para calcular a média
BATCH_SIZES = [1, 8, 32, 128, 512] # Tamanhos de lote para testar

# --- SETUP: CARREGAMENTO DOS MODELOS ---

# 1. Carregar modelo PyTorch
PYTORCH_MODEL_PATH = "quadratic_model.pth"
pytorch_model = QuadraticModel()
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
pytorch_model.eval()

# 2. Carregar modelo ONNX
ONNX_MODEL_PATH = "quadratic_model.onnx"
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = onnx_session.get_inputs()[0].name

print("="*50)
print("Iniciando Benchmark de Inferência: PyTorch vs. ONNX")
print(f"Número de execuções por teste: {NUM_RUNS}")
print("="*50)

# --- LOOP DE BENCHMARKING ---

for batch_size in BATCH_SIZES:
    print(f"\n--- Testando com Lote (Batch Size) = {batch_size} ---")
    
    # Gerar dados de teste aleatórios com o tamanho do lote atual
    dummy_data_np = np.random.randn(batch_size, 1).astype(np.float32)
    dummy_data_torch = torch.from_numpy(dummy_data_np)
    
    # --- WARM-UP (AQUECIMENTO) ---
    # Executa algumas vezes antes para inicializar tudo
    for _ in range(10):
        _ = pytorch_model(dummy_data_torch)
        _ = onnx_session.run(None, {input_name: dummy_data_np})

    # --- BENCHMARK PYTORCH ---
    # Usamos time.perf_counter() para medições de tempo precisas
    start_time = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad(): # Desativa o cálculo de gradientes para máxima velocidade
            _ = pytorch_model(dummy_data_torch)
    end_time = time.perf_counter()
    
    pytorch_total_time = end_time - start_time
    pytorch_time_per_inference = (pytorch_total_time / NUM_RUNS) * 1000 # em milissegundos
    pytorch_inferences_per_sec = NUM_RUNS / pytorch_total_time

    print(f"PyTorch:")
    print(f"\tTempo médio: {pytorch_time_per_inference:.6f} ms/inferência")
    print(f"\tTaxa: {pytorch_inferences_per_sec:.2f} inferências/segundo")

    # --- BENCHMARK ONNX RUNTIME ---
    start_time = time.perf_counter()
    for _ in range(NUM_RUNS):
        _ = onnx_session.run(None, {input_name: dummy_data_np})
    end_time = time.perf_counter()

    onnx_total_time = end_time - start_time
    onnx_time_per_inference = (onnx_total_time / NUM_RUNS) * 1000 # em milissegundos
    onnx_inferences_per_sec = NUM_RUNS / onnx_total_time
    
    print(f"ONNX Runtime:")
    print(f"\tTempo médio: {onnx_time_per_inference:.6f} ms/inferência")
    print(f"\tTaxa: {onnx_inferences_per_sec:.2f} inferências/segundo")
    
    # --- COMPARAÇÃO ---
    if onnx_time_per_inference < pytorch_time_per_inference:
        speedup = pytorch_time_per_inference / onnx_time_per_inference
        print(f"\n\t>> ONNX foi {speedup:.2f}x mais rápido.")
    else:
        speedup = onnx_time_per_inference / pytorch_time_per_inference
        print(f"\n\t>> PyTorch foi {speedup:.2f}x mais rápido.")

print("\n" + "="*50)
print("Benchmark Concluído.")
print("="*50)