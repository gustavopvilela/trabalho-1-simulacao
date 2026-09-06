import pandas as pd
import math
from collections import defaultdict
from sklearn.model_selection import train_test_split

def classifica_naive_bayes (p_prior, p_condicional, exemplo_novo):
    melhor_log_prob = -math.inf
    classe_predita = max(p_prior, key=p_prior.get) # Nunca retorna None, caso não haja resultado, retorna a classe majoritária

    for c in p_prior:
        log_prob = math.log(p_prior[c])

        for a, valor in exemplo_novo.items():
            tabela = p_condicional.get(a)
            if tabela is None: continue

            probs = tabela.get(valor)
            if probs is None: continue

            p = probs[c]
            if p == 0.0:
                log_prob = -math.inf
                break
            log_prob += math.log(p)

        if log_prob > melhor_log_prob:
            melhor_log_prob = log_prob
            classe_predita = c

    return classe_predita

def treina_naive_bayes (dados_treino):
    n = len(dados_treino)

    conta_classe = defaultdict(int)
    for _, c in dados_treino:
        conta_classe[c] += 1

    conta_conjunta = defaultdict(int)
    valores_de = defaultdict(set)
    for exemplo, c in dados_treino:
        for a, v in exemplo.items():
            conta_conjunta[(a, v, c)] += 1
            valores_de[a].add(v)

    p_prior = {c: conta_classe[c] / n for c in conta_classe}
    p_condicional = {}
    for a in valores_de:
        p_condicional[a] = {}
        for v in valores_de[a]:
            p_condicional[a][v] = {
                c: conta_conjunta[(a, v, c)] / conta_classe[c] for c in conta_classe
            }

    return p_prior, p_condicional

if __name__ == '__main__':
    COLUNAS = [
        "classe",
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "el-salvador-aid",
        "religious-groups-in-schools",
        "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missile",
        "immigration",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa"
    ]

    df = pd.read_csv(
        "congressional_voting_records/house-votes-84.data",
        header=None,
        names=COLUNAS,
    )

    dados = df

    treino, teste = train_test_split(
        dados,
        test_size=0.2,
        stratify=dados["classe"],
        random_state=None
    )

    COLS = [c for c in dados.columns if c != "classe"]
    dados_treino = list(zip(treino[COLS].to_dict(orient="records"), treino["classe"]))

    X_teste = teste[COLS].to_dict(orient="records")
    y_teste = teste["classe"].tolist()

    p_prior, p_condicional = treina_naive_bayes(dados_treino)
    preds = [classifica_naive_bayes(p_prior, p_condicional, ex) for ex in X_teste]

    acertos = sum(p == y for p, y in zip(preds, y_teste))
    print(f"Acurácia: {acertos / len(y_teste):.2%}")

    for c in sorted(set(y_teste)):
        total = sum(y == c for y in y_teste)
        certos = sum(p == y for p, y in zip(preds, y_teste) if y == c)
        print(f"{c}: {certos}/{total} = {certos / total:.2%}")