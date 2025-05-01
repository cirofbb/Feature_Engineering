# Projeto de Feature Engineering e Pré-Processamento de Dados

Este projeto em Python explora diversas técnicas de Feature Engineering e pré-processamento de dados aplicadas a diferentes tipos de datasets e problemas de Machine Learning (Regressão e Classificação). O objetivo é demonstrar a aplicação prática dessas técnicas e analisar seu impacto nos resultados dos modelos.

## Conteúdo do Projeto

O notebook Python aborda os seguintes tópicos:

1.  **Discretização com Bins Fixos (Regressão):**
    * Aplicação da técnica de discretização por quantização com bins de largura fixa em uma característica do dataset de regressão de médias salariais.
    * Treinamento de um modelo de Regressão Linear utilizando apenas a característica discretizada.
    * Cálculo da métrica $R^2$ para avaliar o desempenho do modelo.

2.  **Discretização com Bins Variáveis (Regressão):**
    * Aplicação da técnica de discretização por quantização com bins baseados em quantis em outra característica do mesmo dataset de regressão.
    * Treinamento de um modelo de Regressão Linear utilizando apenas a característica discretizada.
    * Cálculo da métrica $R^2$ para avaliar o desempenho do modelo.

3.  **Power Transform (Classificação):**
    * Aplicação da técnica PowerTransform em uma característica do dataset de classificação breast_cancer.
    * Treinamento de um modelo de Regressão Logística utilizando a característica transformada.
    * Avaliação do desempenho do modelo utilizando validação cruzada e as métricas de acurácia e F1 score.

4.  **MinMaxScaler vs. StandardScaler (Classificação):**
    * Aplicação das técnicas MinMaxScaler e StandardScaler em uma característica do dataset de classificação.
    * Explicação detalhada das diferenças entre as duas técnicas de escalonamento e seus casos de uso.

5.  **Cálculo da Norma-L2:**
    * Implementação e explicação do cálculo da norma-L2 (Euclidiana) para uma característica, com a fórmula:
        $$\|x\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$$
    * Discussão sobre a utilidade da normalização L2 no contexto de Machine Learning.

6.  **Processamento de Texto - Lematização, Bag-of-nGrams e TF-IDF (Classificação de Texto):**
    * Utilização do dataset do IMDb.
    * Aplicação da técnica de **lematização** para reduzir as palavras à sua forma base.
    * Implementação da abordagem **Bag-of-nGrams** (com um tamanho de contexto específico).
    * Aplicação da técnica **TF-IDF (Term Frequency-Inverse Document Frequency)** para ponderar a importância das palavras.
    * Avaliação do desempenho de um modelo de classificação utilizando validação cruzada e as métricas de acurácia e F1 score.

7.  **Processamento de Texto - Stemming, Bag-of-Words e TF-IDF (Classificação de Texto):**
    * Utilização do dataset do IMDb.
    * Aplicação da técnica de **stemming** para reduzir as palavras ao seu radical.
    * Implementação da abordagem **Bag-of-Words**.
    * Aplicação da técnica **TF-IDF**.
    * Avaliação do desempenho de um modelo de classificação utilizando validação cruzada e as métricas de acurácia e F1 score.

8.  **Codificação de Variáveis Categóricas (Classificação):**
    * Utilização do dataset Titanic.
    * Aplicação das técnicas de **One-Hot Encoding**, **Dummy Coding** e **Effect Coding** em todas as variáveis categóricas.
    * Avaliação comparativa do desempenho de um modelo de classificação para cada uma das três técnicas de codificação.

9.  **FunctionTransformer para Bin Counting:**
    * Utilização da função `FunctionTransformer` do scikit-learn para transformar uma implementação de Bin Counting em uma função compatível com os pipelines do scikit-learn.
    * Aplicação da transformação a um dataset escolhido.

10. **Análise de Componentes Principais (PCA):**
    * Utilização do Iris Dataset.
    * Aplicação do algoritmo PCA para reduzir a dimensionalidade do dataset.
    * Análise do número de componentes principais (PCs) necessários para representar a variância total dos dados.

11. **Explicação Teórica do PCA:**
    * Explicação detalhada do funcionamento do algoritmo PCA.
    * Definição e interpretação dos **autovalores** e **autovetores** dos componentes principais.

## Como Usar

Para executar este projeto, você precisará ter o Python instalado, juntamente com as seguintes bibliotecas:

* pandas
* scikit-learn
* numpy

Você pode instalar as dependências utilizando o pip:

```bash
pip install pandas scikit-learn numpy
