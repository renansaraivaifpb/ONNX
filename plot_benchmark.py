# plot_benchmark.py
import time
import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
from train import QuadraticModel

# --- PARÂMETROS DO BENCHMARK ---
NUM_RUNS = 500
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] 

# --- SETUP: CARREGAMENTO DOS MODELOS ---
PYTORCH_MODEL_PATH = "quadratic_model.pth"
pytorch_model = QuadraticModel()
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
pytorch_model.eval()

ONNX_MODEL_PATH = "quadratic_model.onnx"
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = onnx_session.get_inputs()[0].name

print("="*50)
print("Coletando dados para o gráfico de benchmark...")
print("="*50)

# --- Listas para armazenar os resultados ---
pytorch_ips_results = [] # IPS = Inferências por Segundo
onnx_ips_results = []

# --- COLETA DE DADOS ---
for batch_size in BATCH_SIZES:
    print(f"Executando benchmark para lote = {batch_size}...")
    dummy_data_np = np.random.randn(batch_size, 1).astype(np.float32)
    dummy_data_torch = torch.from_numpy(dummy_data_np)
    
    # Warm-up
    for _ in range(10):
        _ = pytorch_model(dummy_data_torch)
        _ = onnx_session.run(None, {input_name: dummy_data_np})

    # Benchmark PyTorch
    start_time = time.perf_counter()
    for _ in range(NUM_RUNS):
        with torch.no_grad():
            _ = pytorch_model(dummy_data_torch)
    end_time = time.perf_counter()
    pytorch_total_time = end_time - start_time
    pytorch_ips = NUM_RUNS / pytorch_total_time
    pytorch_ips_results.append(pytorch_ips)

    # Benchmark ONNX Runtime
    start_time = time.perf_counter()
    for _ in range(NUM_RUNS):
        _ = onnx_session.run(None, {input_name: dummy_data_np})
    end_time = time.perf_counter()
    onnx_total_time = end_time - start_time
    onnx_ips = NUM_RUNS / onnx_total_time
    onnx_ips_results.append(onnx_ips)

print("\nColeta de dados concluída. Gerando o gráfico...")

# --- PLOTAGEM DOS RESULTADOS ---
plt.style.use('seaborn-v0_8-whitegrid') # Estilo de gráfico profissional
plt.figure(figsize=(12, 8)) # Tamanho da figura

plt.plot(BATCH_SIZES, pytorch_ips_results, 'o-', label='PyTorch', color='dodgerblue', linewidth=2)
plt.plot(BATCH_SIZES, onnx_ips_results, 's--', label='ONNX Runtime', color='red', linewidth=2)

# --- Detalhes do Gráfico ---
plt.title('Comparativo de Performance: PyTorch vs. ONNX Runtime', fontsize=16, fontweight='bold')
plt.xlabel('Tamanho do Lote (Batch Size)', fontsize=12)
plt.ylabel('Inferências por Segundo (Maior é Melhor)', fontsize=12)
plt.legend(fontsize=12)

# Usando escala logarítmica no eixo X para melhor visualização
# Tamanhos de lote crescem exponencialmente, a escala log os espaça igualmente.
plt.xscale('log') 
# pode também usar escala log no eixo Y se os resultados variarem muito
# plt.yscale('log') 

# Adiciona os valores exatos no eixo X para clareza
plt.xticks(BATCH_SIZES, labels=BATCH_SIZES, rotation=45) 
plt.minorticks_off() # Desliga marcações menores que podem poluir o gráfico com rotação

plt.tight_layout() # Ajusta o layout para não cortar os rótulos

# SALVANDO O PLOT
PLOT_FILENAME = "benchmark_plot.png"
plt.savefig(PLOT_FILENAME, dpi=300) # dpi=300 para alta resolução
print(f"Gráfico de benchmark salvo como '{PLOT_FILENAME}'")

plt.show()