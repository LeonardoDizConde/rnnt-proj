# Imports
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import re
import nltk
import spacy
import string
from collections import Counter
from sklearn.metrics import confusion_matrix
import pickle

# Diretórios
ASSETS_DIR = Path("./new_assets")
MODEL_DIR = Path("./models/lstm")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Labels
LABEL_PREFIX_MAP = {
    'benigno': 0,
    'suspeito': 1,
    'maligno': 2,
}
INDEX_TO_LABEL = {v: k for k, v in LABEL_PREFIX_MAP.items()}
ALL_CLASSIFICATIONS = [
    "malware",
    "phishing",
    "botnet",
    "trojan",
    "ransomware",
    "spam"
]

LABEL_MAP = {}
INDEX_TO_LABEL = {}

label_idx = 0

# Benigno (sem classificação)
LABEL_MAP["benigno"] = label_idx
INDEX_TO_LABEL[label_idx] = "benigno"
label_idx += 1

# Suspeito
for cls in ALL_CLASSIFICATIONS:
    key = f"suspeito_{cls}"
    LABEL_MAP[key] = label_idx
    INDEX_TO_LABEL[label_idx] = key
    label_idx += 1

# Maligno
for cls in ALL_CLASSIFICATIONS:
    key = f"maligno_{cls}"
    LABEL_MAP[key] = label_idx
    INDEX_TO_LABEL[label_idx] = key
    label_idx += 1

def parse_filename(filename: str):
    name = filename[:-4]
    parts = name.split("_")

    if len(parts) < 1:
        return None

    label_str = parts[0]  # benigno, suspeito, maligno

    classification = "outro"
    if len(parts) >= 2 and parts[1]:
        classifications = parts[1].split("-")
        classification = classifications[0] if classifications else "outro"

    # Se benigno, forçar "outro"
    if label_str == "benigno":
        combined_label = "benigno_outro"
    else:
        if classification not in ALL_CLASSIFICATIONS:
            classification = "outro"
        combined_label = f"{label_str}_{classification}"

    label_int = LABEL_MAP.get(combined_label)
    if label_int is None:
        return None

    return label_int

# NLP
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
pontuacao_lista = list(string.punctuation.strip()) + ['...', '“', '”']

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

def clean_text(text: str) -> str:
    tokens = nlp(text)
    tokens = [str(t).lower() for t in tokens if str(t) not in pontuacao_lista]
    tokens = [str(t) for t in tokens if str(t) not in stopwords]
    return " ".join(tokens)

# Parse filename
def parse_filename(filename: str):
    # Remove .txt
    name = filename[:-4]
    parts = name.split("_")

    label_str = parts[0]
    label_int = LABEL_PREFIX_MAP.get(label_str)
    if label_int is None:
        return None

    classifications = []
    ip = None

    if len(parts) == 2:
        if parts[1]:
            # Tem classificações
            classifications = parts[1].split("-")
    elif len(parts) == 3:
        # Se veio com __ (parte 1 vazia), parts[1] == ""
        if parts[1]:
            classifications = parts[1].split("-")
        if parts[2]:
            ip = parts[2].replace("-", ".")
    
    return label_int, classifications, ip

def load_reports() -> Tuple[List[str], List[int]]:
    texts, labels = [], []
    files = list(ASSETS_DIR.glob("*.txt"))
    total_files = len(files)

    for i, file in enumerate(files, start=1):
        parsed = parse_filename(file.name)
        if not parsed:
            print(f"[ERRO] Nome inválido: {file.name}")
            continue

        label, classifications, ip = parsed
        try:
            data = json.loads(file.read_text())
            raw = json.dumps(data)
            classifications_text = " ".join(classifications) if classifications else ""
            full_text = f"{raw} {classifications_text}"
            texts.append(clean_text(full_text))
            labels.append(label)
        except Exception as e:
            print(f"[ERRO] Falha ao processar {file.name}: {e}")

        percent = (i / total_files) * 100
        print(f"\rLoading reports: {percent:.2f} %", end="", flush=True)

    print()
    counts = Counter(labels)
    print(f"[INFO] Distribuição das classes (total de arquivos: {total_files}):")
    for label in range(len(INDEX_TO_LABEL)):
        count = counts.get(label, 0)
        perc = (count / total_files) * 100 if total_files else 0
        print(f"\t{INDEX_TO_LABEL[label]}: {perc:.2f}% ({count})")

    return texts, labels

def train():
    X, y = load_reports()

    if len(set(y)) < 2:
        print("[ERRO] Dados insuficientes.")
        return {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tokenizer.fit_on_texts(X_train)

    with open(MODEL_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    seq_train = tokenizer.texts_to_sequences(X_train)
    seq_test = tokenizer.texts_to_sequences(X_test)

    padded_train = pad_sequences(seq_train, padding='post')
    padded_test = pad_sequences(seq_test, padding='post')

    vocab_size = len(tokenizer.word_index) + 1
    num_classes = len(LABEL_MAP)

    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = to_categorical(y_test, num_classes=num_classes)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64),
        LSTM(128, return_sequences=True),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # model = Sequential([
    #     Embedding(input_dim=vocab_size, output_dim=128),
    #     Bidirectional(LSTM(128, return_sequences=True)),
    #     GlobalMaxPooling1D(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.3),
    #     Dense(13, activation='softmax')
    # ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    # EarlyStopping + Checkpoint
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=MODEL_DIR / "best_model.keras", monitor='val_loss', save_best_only=True)
    ]

    model.fit(
        padded_train, y_train_ohe,
        validation_data=(padded_test, y_test_ohe),
        epochs=35,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    return model, padded_test, y_test, tokenizer

if __name__ == "__main__":
    model, X_test, y_test, tokenizer = train()
    if model:
        print(model.summary())

        # Prever todo o conjunto de teste em batch
        preds = model.predict(X_test, verbose=0)
        y_pred = preds.argmax(axis=-1)

        num_classes = len(LABEL_MAP)

        # Criar matriz de confusão zerada
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # Preencher a matriz de confusão
        for true_label, pred_label in zip(y_test, y_pred):
            confusion_matrix[true_label, pred_label] += 1

        # Plotar matriz
        fig, ax = plt.subplots(figsize=(12,12))
        cax = ax.matshow(confusion_matrix, cmap='Blues')

        plt.title("Matriz de Confusão")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")

        tick_labels = [INDEX_TO_LABEL[i] for i in range(num_classes)]
        plt.xticks(range(num_classes), tick_labels, rotation=90)
        plt.yticks(range(num_classes), tick_labels)

        # Inserir números nas células
        for (i, j), val in np.ndenumerate(confusion_matrix):
            ax.text(j, i, f"{val}", ha="center", va="center", color="black")

        fig.colorbar(cax)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.close()
        print("Gráfico salvo em confusion_matrix.png")
