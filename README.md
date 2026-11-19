# Classificação de flores Iris com Rede Neural MLP

**Disciplina:** Inteligência Artificial  
**Tipo de problema:** classificação multiclasse  
**Algoritmo:** MLP (Perceptron de Múltiplas Camadas), `MLPClassifier` do Scikit-learn  
**Validação:** Hold-out, 80% de treino e 20% de teste

## 1. Objetivo

Treinar uma rede neural para reconhecer a espécie de uma flor Iris usando quatro medidas em centímetros: comprimento e largura da sépala e comprimento e largura da pétala.

As classes são **setosa**, **versicolor** e **virginica**. A base contém **150 registros** (50 de cada classe), com **4 atributos numéricos**. Ela foi originalmente publicada por Fisher e está disponível na UCI:

https://archive.ics.uci.edu/dataset/53/iris

Este programa carrega a versão Iris que acompanha o Scikit-learn (`load_iris`). Ela corresponde essencialmente ao conjunto clássico da UCI, mas possui correções em dois registros documentadas pelo próprio Scikit-learn. Portanto, os resultados podem variar ligeiramente se você usar diretamente o arquivo original do site UCI.

## 2. Como executar

É necessário ter Python 3.10 ou superior e acesso aos pacotes listados em `requirements.txt`.

1. Extraia o ZIP e abra o terminal na pasta `trabalho_mlp_iris`.
2. (Opcional, recomendado) Crie um ambiente virtual:
   - Windows PowerShell: `py -m venv .venv` e `.\.venv\Scripts\Activate.ps1`
   - Linux/macOS: `python3 -m venv .venv` e `source .venv/bin/activate`
3. Instale as dependências:
   - Windows: `py -m pip install -r requirements.txt`
   - Linux/macOS: `python3 -m pip install -r requirements.txt`
4. Execute:
   - Windows: `py trabalho_mlp_iris.py`
   - Linux/macOS: `python3 trabalho_mlp_iris.py`

O programa imprime no terminal os dados da base, o total de registros de treino e teste, as métricas, a matriz de confusão e um exemplo de previsão.

### Classificar uma flor com medidas escolhidas por você

```bash
python trabalho_mlp_iris.py --medidas 5.1 3.5 1.4 0.2
```

As medidas, em centímetros, devem estar nesta ordem: **comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala**.

## 3. Como o código funciona

1. `carregar_e_dividir_dados`: lê a base Iris e faz a divisão Hold-out. `stratify=y` conserva a proporção entre as classes.
2. `criar_e_treinar_mlp`: cria um `Pipeline` com `StandardScaler` e `MLPClassifier`, garantindo que a padronização seja aprendida apenas no treinamento.
3. `avaliar_modelo`: usa as 30 amostras separadas para teste e calcula acurácia, precisão, recall e F1-score, além de um relatório por espécie.
4. `salvar_graficos`: gera a matriz de confusão e a curva da perda do treinamento.
5. `demonstrar_previsao`: mostra uma classificação individual e as probabilidades previstas.

### Arquitetura da rede

```text
4 entradas (medidas da flor)
            |
16 neurônios (camada oculta 1, ReLU)
            |
 8 neurônios (camada oculta 2, ReLU)
            |
 3 saídas (uma classe para cada espécie)
```

A rede utiliza o algoritmo de otimização **Adam** para ajustar os pesos durante o treinamento.

## 4. Métricas utilizadas

- **Acurácia**: proporção de previsões corretas entre todos os exemplos de teste.
- **Precisão**: entre as amostras previstas como uma classe, proporção que realmente pertence àquela classe.
- **Recall**: entre as amostras que realmente pertencem à classe, proporção que foi corretamente identificada.
- **F1-score**: média harmônica entre precisão e recall.
- **Matriz de confusão**: tabela com as classes reais nas linhas e as previstas nas colunas. Valores na diagonal indicam acertos.

Como existem 3 classes, as métricas de precisão, recall e F1-score mostradas no resumo são calculadas com **média macro**: calcula-se a métrica por classe e tira-se uma média simples, dando o mesmo peso a cada espécie.

## 5. Resultado da execução testada

Ambiente usado para verificar o programa: Scikit-learn 1.8.0, NumPy 2.3.5, Matplotlib 3.10.8.

| Métrica | Resultado |
| --- | ---: |
| Acurácia | 93,33% |
| Precisão (macro) | 93,33% |
| Recall (macro) | 93,33% |
| F1-score (macro) | 93,33% |

Matriz de confusão (ordem setosa / versicolor / virginica):

```text
[[10, 0, 0],
 [ 0, 9, 1],
 [ 0, 1, 9]]
```

Isso corresponde a 28 acertos em 30 amostras de teste. As diferenças entre espécies próximas podem gerar erros. O valor obtido descreve somente esta separação de dados, não um desempenho garantido em outros conjuntos.

## 6. Arquivos gerados

Após executar o código, a pasta `resultados/` contém:

- `metricas.txt`: números e relatório completo por classe.
- `matriz_confusao.png`: gráfico das previsões corretas e incorretas.
- `curva_perda.png`: evolução da perda durante o treinamento.

## 7. Entrega da atividade

A entrega da atividade contém o código-fonte (`trabalho_mlp_iris.py`) e o vídeo de apresentação, no qual explico o projeto, a implementação da rede neural MLP, a execução do programa e as métricas de avaliação.

**Vídeo explicativo do projeto (YouTube):** https://www.youtube.com/watch?v=ihF7WpAv1-0

## Referências

- Fisher, R. A. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76
- Scikit-learn. `load_iris`: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
- Scikit-learn. `MLPClassifier`: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
