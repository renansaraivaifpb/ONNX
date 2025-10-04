# Otimização de Modelos PyTorch para Inferência de Borda com ONNX

Este repositório apresenta um pipeline completo para demonstrar e quantificar os benefícios da otimização de modelos de Deep Learning (treinados em PyTorch) para inferência em ambientes com recursos limitados, utilizando o formato **ONNX (Open Neural Network Exchange)** e seu motor de execução **ONNX Runtime**.

## Visão Geral do Projeto

A motivação para este projeto surgiu de um desafio real: a necessidade de implantar uma complexa rede U-Net para segmentação semântica em uma **Raspberry Pi 3**. Para abordar a questão da performance e portabilidade em hardware restrito, este repositório explora a jornada de um modelo PyTorch desde o treinamento até a inferência otimizada, comparando diretamente o desempenho do PyTorch nativo com o ONNX Runtime.

O projeto cobre as seguintes etapas:
1.  **Treinamento de um Modelo PyTorch:** Um modelo simples, mas com capacidade não-linear, é treinado em PyTorch.
2.  **Exportação para ONNX:** O modelo treinado é convertido para o formato universal ONNX.
3.  **Comparação de Inferência:** Validação funcional das predições entre o modelo PyTorch e o modelo ONNX.
4.  **Benchmark de Performance:** Medição rigorosa da velocidade de inferência (Inferências por Segundo - IPS) de ambos os modelos em diferentes tamanhos de lote.
5.  **Visualização dos Resultados:** Geração de gráficos para análise clara da performance.
