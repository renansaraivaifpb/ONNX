# export_to_onnx.py
import torch
# Importa a nova arquitetura do nosso arquivo de treino
from train import QuadraticModel 

# 1. Instanciar a arquitetura correta do modelo
model = QuadraticModel()

# 2. Carregar os pesos treinados do novo modelo
MODEL_PATH = "quadratic_model.pth"
ONNX_PATH = "quadratic_model.onnx"
model.load_state_dict(torch.load(MODEL_PATH))

# 3. Colocar o modelo em modo de inferência
model.eval()

# 4. Criar uma entrada de exemplo
dummy_input = torch.randn(1, 1) 

# 5. Exportar o modelo
torch.onnx.export(
    model,                          # O modelo a ser exportado
    dummy_input,                    # Uma entrada de exemplo
    ONNX_PATH,                      # Onde salvar o arquivo .onnx
    export_params=True,             # Armazena os pesos treinados no arquivo
    opset_version=11,               # Versão do ONNX opset; 11 é bastante estável
    do_constant_folding=True,       # Executa otimizações de dobra de constantes
    input_names=['input'],          # Nome da camada de entrada
    output_names=['output'],        # Nome da camada de saída
    dynamic_axes={'input': {0: 'batch_size'}, # Permite que o tamanho do batch seja dinâmico
                  'output': {0: 'batch_size'}}
)

print(f"Modelo exportado para o formato ONNX em {ONNX_PATH}")