# Imports
import json
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
import spacy
import string
import os
from collections import Counter

# Diretórios
ASSETS_DIR = Path("./assets")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# Palavras-chave
KEYWORDS_MALICIOUS = {"malicious", "malware", "trojan", "phishing", "botnet", "miner"}
KEYWORDS_SUSPICIOUS = {"suspicious", "spam", "unrated", "risk", "unknown"}
RULE_WEIGHTS = {**{k: 1 for k in KEYWORDS_SUSPICIOUS}, **{k: 2 for k in KEYWORDS_MALICIOUS}}

# Modelos
PIPELINES = {
    "bow_nb": Pipeline([
        ("vect", CountVectorizer()),
        ("clf", MultinomialNB()),
    ]),
    "tfidf_svc": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC()),
    ]),
}

# NLP
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
pontuacao_lista = list(string.punctuation.strip()) + ['...', '“', '”']

# Limpeza de texto
def clean_text(text: str) -> str:
    tokens = nlp(text)
    tokens = [str(t).lower() for t in tokens if str(t) not in pontuacao_lista]
    tokens = [str(t) for t in tokens if str(t) not in stopwords]
    return " ".join(tokens)

# Normalizador universal de relatórios
def normalize_report(report: dict) -> dict:
    if "full" in report and "attributes" in report["full"]:
        return report
    if "data" in report and "attributes" in report["data"]:
        return {"full": report["data"]}
    if "attributes" in report:
        return {"full": report}
    raise ValueError("Formato de relatório não reconhecido.")

# Carregamento dos relatórios
def load_reports() -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for file in ASSETS_DIR.glob("*.txt"):
        try:
            data = json.loads(file.read_text())
            data = normalize_report(data)

            last_analysis = data['full'].get('attributes', {}).get('last_analysis_results', {})
            malicious_count = sum(
                1 for val in last_analysis.values()
                if val.get('category') in KEYWORDS_MALICIOUS or val.get('result') in KEYWORDS_MALICIOUS
            )

            label = (
                "malicioso" if malicious_count >= 5 else
                "suspeito" if 1 < malicious_count < 5 else
                "benigno"
            )

            raw = json.dumps(data)
            texts.append(clean_text(raw))
            labels.append(label)
        except Exception as e:
            print(f"[ERRO] ao processar {file.name}: {e}")
    return texts, labels

# Regras manuais baseadas em palavras-chave
def rule_based_predict(text: str) -> str:
    score = sum(RULE_WEIGHTS.get(tok, 0) for tok in text.split())
    if score >= 4:
        return "malicioso"
    if score >= 1:
        return "suspeito"
    return "benigno"

# Combinação dos classificadores
def ensemble_predict(text: str, models) -> str:
    preds = [m.predict([text])[0] for m in models.values()]
    preds.append(rule_based_predict(text))
    return max(set(preds), key=preds.count)

# Geração de comentários
def generate_comment(label: str, raw_json: dict) -> str:
    raw_json = normalize_report(raw_json)
    attr = raw_json["full"]["attributes"]
    ip = raw_json["full"].get("id", "N/A")
    stats = attr.get("last_analysis_stats", {})
    tags = ", ".join(attr.get("tags", []))
    country = attr.get("country", "Desconhecido")
    as_owner = attr.get("as_owner", "Desconhecido")
    reputation = attr.get("reputation", "N/A")
    mal = stats.get("malicious", 0)
    susp = stats.get("suspicious", 0)
    harmless = stats.get("harmless", 0)

    if label == "malicioso":
        return (
            f"Indicador classificado como malicioso com {mal} detecções confirmadas. "
            f"O IP {ip} pertence ao ASN '{as_owner}' ({country}), com reputação {reputation}. "
            f"Tags: {tags}. Recomendado bloqueio e investigação."
        )
    elif label == "suspeito":
        return (
            f"Indicador suspeito com {susp} alertas. IP {ip}, ASN '{as_owner}' ({country}). "
            f"Monitoramento contínuo é recomendado."
        )
    else:
        return (
            f"Indicador benigno com {harmless} detecções limpas. IP {ip}, ASN '{as_owner}' ({country}). "
            f"Sem sinais de risco atuais."
        )

# Treinamento dos modelos
def train():
    X, y = load_reports()
    print(f"[INFO] Distribuição das classes: {Counter(y)}")

    if len(set(y)) < 2:
        print("[ERRO] Dados insuficientes: é necessário ao menos duas classes diferentes.")
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {}
    for name, pipe in PIPELINES.items():
        try:
            pipe.fit(X_train, y_train)
            joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
            preds = pipe.predict(X_test)
            print(f"\n*** {name} ***")
            print(classification_report(y_test, preds, zero_division=0))
            models[name] = pipe
        except Exception as e:
            print(f"[ERRO] Falha ao treinar {name}: {e}")

    print("✅ Treinamento concluído.")
    return models

# Predição com novo modelo
def predict(path: Path, models):
    try:
        raw = json.loads(path.read_text())
        raw = normalize_report(raw)
        text = clean_text(json.dumps(raw))
        label = ensemble_predict(text, models)
        comment = generate_comment(label, raw)
        print(f"📄 {path.name} -> {label.upper()}")
        print(comment)
    except Exception as e:
        print(f"[ERRO] ao processar {path.name}: {e}")

# Execução principal
if __name__ == "__main__":
    models = train()
    if models:
        print("\n🔍 Executando predições nos relatórios:")
        for file in ASSETS_DIR.glob("*.txt"):
            predict(file, models)

        # Exporta o modelo padrão para integração com webhook
        joblib.dump(models.get("tfidf_svc"), "modelo_pln.pkl")