"""Trabalho de Aprendizado supervisionado: José Guilherme e Bruno Henrique Frota"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PASTA_RESULTADOS = Path(__file__).resolve().parent / "resultados"


def carregar_e_dividir_dados():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,  
    )
    return iris, X_treino, X_teste, y_treino, y_teste


def criar_e_treinar_mlp(X_treino, y_treino):
    modelo = Pipeline(
        steps=[
            ("padronizacao", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(16, 8),  # 16 e 8 neuronios nas camadas ocultas.
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.01,
                    alpha=0.001,
                    max_iter=1500,
                    random_state=42,
                ),
            ),
        ]
    )
    modelo.fit(X_treino, y_treino)
    return modelo


def avaliar_modelo(modelo, X_teste, y_teste, nomes_classes):
    previsoes = modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, previsoes)
    precisao = precision_score(y_teste, previsoes, average="macro", zero_division=0)
    recall = recall_score(y_teste, previsoes, average="macro", zero_division=0)
    f1 = f1_score(y_teste, previsoes, average="macro", zero_division=0)

    relatorio = classification_report(
        y_teste,
        previsoes,
        target_names=nomes_classes,
        zero_division=0,
        digits=4,
    )
    texto = (
        "=== RESULTADOS DO TESTE (30 FLORES) ===\n"
        f"Acuracia:         {acuracia:.4f} ({acuracia:.2%})\n"
        f"Precisao (macro): {precisao:.4f}\n"
        f"Recall (macro):   {recall:.4f}\n"
        f"F1-score (macro): {f1:.4f}\n\n"
        "Relatorio por especie:\n"
        f"{relatorio}"
    )
    print("\n" + texto)
    (PASTA_RESULTADOS / "metricas.txt").write_text(texto, encoding="utf-8")
    return previsoes


def salvar_graficos(modelo, y_teste, previsoes, nomes_classes):
    matriz = confusion_matrix(y_teste, previsoes, labels=[0, 1, 2])
    figura, eixo = plt.subplots(figsize=(7, 5))
    visualizacao = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=nomes_classes,
    )
    visualizacao.plot(ax=eixo, cmap="Blues", colorbar=False, values_format="d")
    eixo.set_title("Matriz de confusao - MLP no conjunto de teste")
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")
    figura.tight_layout()
    figura.savefig(PASTA_RESULTADOS / "matriz_confusao.png", dpi=160)
    plt.close(figura)

    perdas = modelo.named_steps["mlp"].loss_curve_
    figura, eixo = plt.subplots(figsize=(7, 4))
    eixo.plot(np.arange(1, len(perdas) + 1), perdas)
    eixo.set_title("Curva de perda durante o treinamento")
    eixo.set_xlabel("Iteracao")
    eixo.set_ylabel("Loss (perda)")
    eixo.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(PASTA_RESULTADOS / "curva_perda.png", dpi=160)
    plt.close(figura)

    print("Matriz de confusao (linhas = classes reais; colunas = previstas):")
    print(matriz)
    print(f"\nGraficos e metricas salvos em: {PASTA_RESULTADOS}")


def demonstrar_previsao(modelo, iris, X_teste, y_teste, medidas=None):
    if medidas is None:
        flor = X_teste[0]
        especie_real = iris.target_names[y_teste[0]]
        print("\n=== PREVISAO DE UMA FLOR DO CONJUNTO DE TESTE ===")
        print(f"Especie real: {especie_real}")
    else:
        flor = np.asarray(medidas, dtype=float)
        print("\n=== PREVISAO COM MEDIDAS INFORMADAS ===")

    print("Medidas [sepala comprimento, sepala largura, petala comprimento, petala largura]:")
    print(f"{flor.tolist()} cm")
    classe_prevista = modelo.predict(flor.reshape(1, -1))[0]
    probabilidades = modelo.predict_proba(flor.reshape(1, -1))[0]
    print(f"Especie prevista: {iris.target_names[classe_prevista]}")
    for especie, prob in zip(iris.target_names, probabilidades):
        print(f"  {especie:12s}: {prob:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Classificacao de flores Iris com MLP.")
    parser.add_argument(
        "--medidas",
        type=float,
        nargs=4,
        metavar=("SEP_COMP", "SEP_LARG", "PET_COMP", "PET_LARG"),
        help="Quatro medidas em cm para classificar uma flor manualmente.",
    )
    argumentos = parser.parse_args()

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    iris, X_treino, X_teste, y_treino, y_teste = carregar_e_dividir_dados()
    print("=== BASE DE DADOS IRIS ===")
    print(f"Total de registros: {len(iris.data)}")
    print(f"Caracteristicas por flor: {len(iris.feature_names)}")
    print(f"Especies: {', '.join(iris.target_names)}")
    print(f"Treinamento: {len(X_treino)} flores (80%)")
    print(f"Teste: {len(X_teste)} flores (20%)")
    print("Arquitetura da MLP: 4 entradas -> 16 neuronios -> 8 neuronios -> 3 saidas")

    modelo = criar_e_treinar_mlp(X_treino, y_treino)
    print(f"Iteracoes de treinamento: {modelo.named_steps['mlp'].n_iter_}")
    previsoes = avaliar_modelo(modelo, X_teste, y_teste, iris.target_names)
    salvar_graficos(modelo, y_teste, previsoes, iris.target_names)
    demonstrar_previsao(modelo, iris, X_teste, y_teste, argumentos.medidas)


if __name__ == "__main__":
    main()
