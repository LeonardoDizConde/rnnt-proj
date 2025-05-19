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

# Definições
# Path
ASSETS_DIR = Path("./assets")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# Keywords
KEYWORDS_MALICIOUS = {"malicious", "malware", "trojan", "phishing", "botnet", "miner"}
KEYWORDS_SUSPICIOUS = {"suspicious", "spam", "unrated", "risk", "unknown"}
RULE_WEIGHTS = {**{k: 1 for k in KEYWORDS_SUSPICIOUS}, **{k: 2 for k in KEYWORDS_MALICIOUS}}

# ML pipeline
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

# Carrega modelos
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

# Utils
pontuacao_lista = list(string.punctuation.strip()) + ['...', '“', '”']

# Lipeza de texto

pontuacao_lista = list(string.punctuation.strip()) + ['...', '“', '”']

def clean_text(text: str) -> str:
    tokens = nlp(text)
    tokens = [str(t).lower() for t in tokens if str(t) not in pontuacao_lista]
    tokens = [str(t) for t in tokens if str(t) not in stopwords]
    return " ".join(tokens)

# Carrega os relatórios e faz um pré-julgamento do relatório
def load_reports() -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for file in ASSETS_DIR.glob("*.txt"):
        data = json.loads(file.read_text())
        last_analysis_results = data['full']['attributes']['last_analysis_results']
        malicious_count = 0
        for _, value in last_analysis_results.items():
            if value.get('category') in KEYWORDS_MALICIOUS or value.get('result') in KEYWORDS_MALICIOUS:
                malicious_count += 1
        label = (
            "malicioso" if malicious_count >= 5 else
            "suspeito" if 1 < malicious_count < 5 else
            "benigno"
        )
        raw = json.dumps(data)
        texts.append(clean_text(raw))
        labels.append(label)
    return texts, labels

# Aplica regra de peso sob os relatórios
def rule_based_predict(text: str) -> str:
    score = sum(RULE_WEIGHTS.get(tok, 0) for tok in text.split())
    if score >= 4:
        return "malicioso"
    if score >= 1:
        return "suspeito"
    return "benigno"

# Classifica os relatórios
def ensemble_predict(text: str, models) -> str:
    preds = [m.predict([text])[0] for m in models.values()]
    preds.append(rule_based_predict(text))
    # Voto por maioria
    return max(set(preds), key=preds.count)

# Gera comentario com ba
def generate_comment(label: str, raw_json: dict) -> str:
    stats = raw_json["full"]["attributes"].get("last_analysis_stats", {})
    mal = stats.get("malicious", 0)
    susp = stats.get("suspicious", 0)
    if label == "malicioso":
        return f"Indicador classificado como malicioso pois {mal} mecanismos antivírus o marcaram como malicioso."
    if label == "suspeito":
        return f"Indicador classificado como suspeito com {susp} detecções suspeitas."
    return "Indicador classificado como benigno; nenhuma detecção relevante encontrada."

# Funções principais
def train():
    X, y = load_reports()
    # Garante que nenhuma classe tenha menos de 2 amostras antes de usar stratify
    from collections import Counter
    counts = Counter(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    for name, pipe in PIPELINES.items():
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
        preds = pipe.predict(X_test)
        print(f"*** {name} ***")
        print(classification_report(y_test, preds, zero_division=0))
    print("Treino concluído.")

    # carrega modelos treinados para uso imediato após o treino
    return {name: joblib.load(MODEL_DIR / f"{name}.joblib") for name in PIPELINES}


def predict(path: Path, models):
    raw = json.loads(path.read_text())
    text = clean_text(json.dumps(raw))
    label = ensemble_predict(text, models)
    comment = generate_comment(label, raw)
    print(json.dumps({"label": label, "comment": comment}, ensure_ascii=False, indent=2))

models = train()

predict_path = './assets'
for filename in os.listdir(predict_path):
  print(f"Report for {filename}:")
  file = Path(os.path.join(predict_path, filename))
  predict(file, models)