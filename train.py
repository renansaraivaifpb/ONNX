# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 1. Definição do Modelo (agora mais robusto)
# Este modelo tem uma camada oculta para aprender relações não-lineares.
class QuadraticModel(nn.Module):
    def __init__(self):
        super(QuadraticModel, self).__init__()
        self.layer1 = nn.Linear(in_features=1, out_features=64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# O código executável agora fica dentro deste bloco
if __name__ == "__main__":
    # 2. Geração de Dados (Dataset Quadrático)
    # A função real que queremos que o modelo aprenda: y = 0.5x² - x - 2
    true_function = lambda x: 0.5 * x**2 - x - 2
    
    # Gera 200 pontos de dados com um pouco de ruído
    X = torch.randn(200, 1) * 10
    y = true_function(X) + torch.randn(200, 1) * 3 # Adicionando ruído

    # 3. Instanciando o Modelo, a Função de Custo e o Otimizador
    model = QuadraticModel()
    criterion = nn.MSELoss()
    # Usaremos o otimizador Adam, que costuma ser mais eficiente para modelos mais complexos
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Loop de Treinamento
    epochs = 1000 # Aumentamos as épocas para aprender a função mais complexa
    for epoch in range(epochs):
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 5. Salvando o Modelo Treinado
    MODEL_PATH = "quadratic_model.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModelo treinado e salvo em {MODEL_PATH}")

    # 6. Visualização e SALVAMENTO do Gráfico de Treinamento
    model.eval()
    with torch.no_grad():
        # Gera uma linha suave de pontos para plotar a curva aprendida
        x_plot = torch.linspace(-10, 10, 100).view(-1, 1)
        y_plot_pred = model(x_plot)

    plt.figure(figsize=(12, 7))
    plt.scatter(X.numpy(), y.numpy(), label='Dados de Treinamento (com ruído)', alpha=0.5, color='blue')
    plt.plot(x_plot.numpy(), true_function(x_plot).numpy(), 'g--', label='Função Verdadeira', linewidth=2)
    plt.plot(x_plot.numpy(), y_plot_pred.numpy(), 'r-', label='Curva Aprendida pelo Modelo', linewidth=3)
    plt.title('Resultados do Treinamento - Modelo Quadrático')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    # SALVANDO O PLOT EM UM ARQUIVO
    PLOT_FILENAME = "training_results.png"
    plt.savefig(PLOT_FILENAME)
    print(f"Gráfico de treinamento salvo como '{PLOT_FILENAME}'")
    
    plt.show() # Opcional: ainda mostra o gráfico na tela