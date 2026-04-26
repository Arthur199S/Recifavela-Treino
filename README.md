# Recifavela-Treino

Treinamento de modelos de Inteligência Artificial para o projeto **Recifavela** — Grupo 1.

Projeto voltado para classificação de resíduos recicláveis, com foco inicial na detecção de materiais **PET** e **NOT_PET**, utilizando visão computacional e aprendizado profundo.

## Objetivo

Desenvolver e treinar modelos de classificação de imagens para auxiliar soluções do projeto Recifavela, explorando:

* Classificação binária (PET / NOT_PET)
* Experimentação com datasets sintéticos e reais
* Técnicas de data augmentation para melhorar generalização
* Fine-tuning de modelos pré-treinados
* Avaliação com métricas como:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix

## Dataset

Dataset base utilizado:

https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset

Baixar e colocar em:

```bash id="a1x9d2"
data/
```

Além do dataset base, o projeto inclui curadoria e complementação manual de imagens para reduzir confusões entre classes e melhorar robustez do modelo.

## Estrutura do Projeto

```bash id="m2p7q4"
Recifavela-Treino/
├── data/
│   ├── PET/
│   └── NOT_PET/
├── src/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── imag-test-certos/
└── README.md
```

## Treinamento

O treinamento utiliza fine-tuning em redes convolucionais, com estratégias como:

* Congelamento e descongelamento progressivo de camadas
* Augmentations para melhorar generalização
* Salvamento do melhor modelo (best_model)
* Validação por métricas e análise de erros

Exemplo de execução:

```bash id="b8k4n1"
python src/train.py
```

## Avaliação

Para avaliar o modelo treinado:

```bash id="c6r5t8"
python src/evaluate.py
```

Exemplo de métricas obtidas em experimentos:

```text id="j3u9w6"
Accuracy: 99%+
F1-score PET: ~97%
```

## Tecnologias

* Python
* PyTorch
* Torchvision
* PIL
* Scikit-learn

## Observações

* O dataset não está versionado no repositório por tamanho.
* Parte do projeto envolve experimentação contínua com novos dados para reduzir falsos positivos e falsos negativos.
* O foco é melhorar a identificação de embalagens PET em cenários variados (cores, formatos, fundos e iluminação).

## Projeto

Projeto desenvolvido no contexto do **Recifavela**, Grupo 1.

## Imagens do ultimo treinamento e do melhor modelo
<img width="1920" height="1032" alt="Captura de tela 2026-04-26 040404" src="https://github.com/user-attachments/assets/0127eaad-50f8-41ff-9744-5296e31332e5" />
<img width="1920" height="1032" alt="Captura de tela 2026-04-26 040327" src="https://github.com/user-attachments/assets/a7b3d936-7a3b-4c00-bb65-d9cdd4047626" />
