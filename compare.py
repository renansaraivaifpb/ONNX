# compare_inference.py
import torch
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

# Importa a arquitetura do modelo para recriar o objeto PyTorch
from train import QuadraticModel

# --- CARREGAMENTO DOS MODELOS ---

# 1. Carregar o modelo PyTorch
PYTORCH_MODEL_PATH = "quadratic_model.pth"
pytorch_model = QuadraticModel()
pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))
pytorch_model.eval() # Essencial para desativar camadas como Dropout/BatchNorm

# 2. Carregar o modelo ONNX
ONNX_MODEL_PATH = "quadratic_model.onnx"
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = onnx_session.get_inputs()[0].name

# --- PREPARAÇÃO DOS DADOS DE TESTE ---

# Criamos um conjunto de dados suave e ordenado para plotar as curvas de predição
# Estes dados NÃO foram usados no treinamento, testando a generalização.
x_test_np = np.linspace(-12, 12, 200, dtype=np.float32).reshape(-1, 1)

# --- EXECUÇÃO DA INFERÊNCIA ---

# 3. Inferência com PyTorch
# PyTorch precisa de tensores como entrada
x_test_torch = torch.from_numpy(x_test_np)
with torch.no_grad():
    pytorch_preds = pytorch_model(x_test_torch).numpy()

# 4. Inferência com ONNX Runtime
# ONNX Runtime precisa de arrays NumPy como entrada
onnx_preds = onnx_session.run(None, {input_name: x_test_np})[0]

# --- COMPARAÇÃO E VISUALIZAÇÃO ---

# 5. Verificar a diferença numérica
# A diferença deve ser extremamente pequena (próxima de zero)
difference = np.abs(pytorch_preds - onnx_preds).max()
print(f"Diferença máxima entre as predições PyTorch e ONNX: {difference}")
if difference < 1e-6:
    print("Validação bem-sucedida: As saídas dos modelos são consistentes!")
else:
    print("Atenção: As saídas dos modelos apresentam divergência significativa.")

# 6. Plotar e salvar a comparação
plt.figure(figsize=(12, 7))
true_function = lambda x: 0.5 * x**2 - x - 2 # A mesma função do treino para referência
plt.plot(x_test_np, true_function(x_test_np), 'g--', label='Função Verdadeira', linewidth=2)
plt.plot(x_test_np, pytorch_preds, 'r-', label='Predição PyTorch', linewidth=4, alpha=0.8)
plt.plot(x_test_np, onnx_preds, 'b:', label='Predição ONNX', linewidth=4, alpha=0.8) # Linha pontilhada azul
plt.title('Comparação de Inferência: PyTorch vs. ONNX Runtime')
plt.xlabel('Entrada (x)')
plt.ylabel('Saída (y)')
plt.legend()
plt.grid(True)

# SALVANDO O PLOT
PLOT_FILENAME = "inference_comparison.png"
plt.savefig(PLOT_FILENAME)
print(f"\nGráfico de comparação salvo como '{PLOT_FILENAME}'")

plt.show()