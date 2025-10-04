# inference.py
import onnxruntime as ort
import numpy as np

# 1. Carregar o modelo ONNX e criar uma sessão de inferência
ONNX_PATH = "simple_model.onnx"
session = ort.InferenceSession(ONNX_PATH)

# 2. Obter os nomes das camadas de entrada e saída (opcional, mas bom para verificação)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Nome da Entrada: {input_name}")
print(f"Nome da Saída: {output_name}")

# 3. Preparar os dados de entrada
# O ONNX Runtime espera arrays NumPy.
# Vamos testar com múltiplos valores.
# O tipo de dado (ex: np.float32) deve ser o mesmo usado no treinamento.
input_data = np.array([[5.0], [10.0], [-3.0]], dtype=np.float32)

# 4. Realizar a inferência
# O método 'run' retorna uma lista de arrays NumPy.
result = session.run([output_name], {input_name: input_data})

# 5. Processar e exibir os resultados
predictions = result[0]
print("\n--- Resultados da Inferência ---")
for i, data in enumerate(input_data):
    print(f"Entrada: {data[0]:.2f} -> Predição: {predictions[i][0]:.4f}")