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
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight

# Diretórios
ASSETS_DIR = Path("./assets")
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# Palavras-chave
KEYWORDS_MALICIOUS = {"malicious", "malware", "trojan", "phishing", "botnet", "miner"}
KEYWORDS_SUSPICIOUS = {"suspicious", "spam", "unrated", "risk", "unknown"}
RULE_WEIGHTS = {**{k: 1 for k in KEYWORDS_SUSPICIOUS}, **{k: 2 for k in KEYWORDS_MALICIOUS}}
LABEL_PREFIX_MAP = {
    'benigno': 1,
    'malicioso': 2,
    'suspeito': 3
}

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

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


# Carregamento dos relatórios
def load_reports() -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    files = list(ASSETS_DIR.glob("*.txt"))
    total_files = len(files)
    
    for i, file in enumerate(files, start=1):
        try:
            data = json.loads(file.read_text())
            label = next((val for prefix, val in LABEL_PREFIX_MAP.items() if file.name.startswith(prefix)), 0)
            raw = json.dumps(data)
            texts.append(clean_text(raw))
            labels.append(label)
        except Exception as e:
            print(f"[ERRO] ao processar {file.name}: {e}")
        
        percent = (i / total_files) * 100
        print(f"\rLoading reports: {percent:.2f} %", end="", flush=True)
    
    print()  # nova linha após terminar o loading
    
    counts = Counter(labels)
    print("[INFO] Distribuição das classes:")
    
    # Para inverter LABEL_PREFIX_MAP de valor para nome (label -> prefix)
    inv_label_map = {v: k for k, v in LABEL_PREFIX_MAP.items()}
    
    for label, count in counts.items():
        name = inv_label_map.get(label, f"label_{label}")
        perc = (count / total_files) * 100
        print(f"\t{name}: {perc:.2f} % -> {count} arquivos")
    
    return texts, labels


# Treinamento dos modelos
def train():
    X, y = load_reports()

    if len(set(y)) < 2:
        print("[ERRO] Dados insuficientes: é necessário ao menos duas classes diferentes.")
        return {}
    
    # Mapear os labels para índices contínuos
    label_set = sorted(set(y))
    label_to_index = {label: idx for idx, label in enumerate(label_set)}
    y = [label_to_index[label] for label in y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    
    tokenizer.fit_on_texts(X_train)

    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    padded_train = pad_sequences(sequences_train, padding='post')
    padded_test = pad_sequences(sequences_test, padding='post')

    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(label_set)

    # Convertendo y_train e y_test para one-hot encoding
    y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    model.fit(
        padded_train, y_train_ohe,
        validation_data=(padded_test, y_test_ohe),
        epochs=35,
        verbose=1,
        class_weight=class_weight_dict
    )

    # Se quiser manter os rótulos originais para uso futuro:
    return model, label_to_index

# Execução principal

if __name__ == "__main__":
    model, label_to_index = train()
    if model:
        print(model.summary())
        print("\n🔍 Executando predições nos relatórios:")

        index_to_label = {idx: label for label, idx in label_to_index.items()}
        num_classes = len(index_to_label)

        # Inicializa matriz 3x3
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        for file in ASSETS_DIR.glob("*.txt"):
            try:
                data = json.loads(file.read_text())
                raw = json.dumps(data)
                cleaned = clean_text(raw)

                if not cleaned.strip():
                    continue

                seq = tokenizer.texts_to_sequences([cleaned])[0]
                seq = [token for token in seq if token is not None]

                if not seq:
                    continue

                padded = pad_sequences([seq], padding='post')

                preds = model.predict(padded, verbose=0)
                pred_class_idx = preds.argmax(axis=-1)[0]

                # Recupera label verdadeiro a partir do prefixo
                true_label = next((val for prefix, val in LABEL_PREFIX_MAP.items() if file.name.startswith(prefix)), 0)
                true_idx = label_to_index[true_label]

                confusion_matrix[true_idx, pred_class_idx] += 1

            except Exception as e:
                print(f"{file.name} -> [ERRO] Falha na predição: {e}")

        # Mostra matriz resultante
        print("\n📊 Matriz de Confusão (linhas=verdadeiro, colunas=previsto):")
        print(confusion_matrix)

        # Opcional: nomes das classes bonitinhos
        print("\nClasses (índices):")
        for idx, label in index_to_label.items():
            print(f"{idx} -> {label} -> {LABEL_PREFIX_MAP[label]}")
        