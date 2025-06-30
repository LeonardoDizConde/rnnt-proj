import requests
import json
import time
from pathlib import Path
import numpy as np
import pickle
import spacy
import nltk
import string
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configurações
VIRUSTOTAL_API_KEY = "b0ff5308c877a949a81bcf43913c0c52098fe915e88e0b9fab9ffd934d9ade65"
HEADERS = {"x-apikey": VIRUSTOTAL_API_KEY}
ASSETS_DIR = Path("./assets")
ASSETS_DIR.mkdir(exist_ok=True)

# Labels - ajuste conforme seu treino
LABEL_PREFIX_MAP = {
    'benigno': 0,
    'suspeito': 1,
    'malicioso': 2,
}

MODELOS_PLN = {
    "bow_nb": joblib.load("./models/bow_nb.joblib"),
    "tfidf_svc": joblib.load("./models/tfidf_svc.joblib")
}

LABEL_INT_TO_STR = {v: k for k, v in LABEL_PREFIX_MAP.items()}

# NLP
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
pontuacao_lista = list(string.punctuation.strip()) + ['...', '“', '”']

# Carregar modelo treinado
model = load_model("models/lstm/best_model.keras")

# Carregar tokenizer treinado
with open("./models/lstm/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Função de limpeza usada no treino
def clean_text(text: str) -> str:
    tokens = nlp(text)
    tokens = [str(t).lower() for t in tokens if str(t) not in pontuacao_lista]
    tokens = [str(t) for t in tokens if str(t) not in stopwords]
    return " ".join(tokens)

# Consulta VirusTotal por IP
def consultar_virustotal(ip):
    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"[ERRO] IP {ip} - Status {response.status_code}")
        return None
    
def classificar_pln(text, models):
    def rule_based_predict(text):
        keywords_malicious = {"malicious", "malware", "trojan", "phishing", "botnet", "miner"}
        keywords_suspicious = {"suspicious", "spam", "unrated", "risk", "unknown"}
        rule_weights = {**{k: 1 for k in keywords_suspicious}, **{k: 2 for k in keywords_malicious}}

        score = sum(rule_weights.get(tok, 0) for tok in text.split())
        if score >= 4:
            return "malicioso"
        elif score >= 1:
            return "suspeito"
        return "benigno"

    preds = [m.predict([text])[0] for m in models.values()]
    preds.append(rule_based_predict(text))
    return max(set(preds), key=preds.count)

# Classificar usando LSTM treinado
def classificar(relatorio):
    try:
        raw = json.dumps(relatorio)
        resultados = relatorio["data"]["attributes"].get("last_analysis_results", {})
        detectados = [dados["engine_name"] for dados in resultados.values() if dados["category"] == "malicious"]
        classifications_text = " ".join(detectados) if detectados else ""
        full_text = f"{raw} {classifications_text}"
        texto_limpo = clean_text(full_text)
        sequencia = tokenizer.texts_to_sequences([texto_limpo])
        X_input = pad_sequences(sequencia, padding='post')
        pred = model.predict(X_input)
        classe = int(np.argmax(pred, axis=1)[0])

        classificacao = LABEL_INT_TO_STR.get(classe, "desconhecido")

        return {
            "classificacao": classificacao,
            "tipos_detectados": detectados
        }

    except Exception as e:
        print(f"[ERRO] Erro ao classificar: {e}")
        return {
            "classificacao": "erro",
            "tipos_detectados": []
        }

# Coletar relatórios e salvar
def coletar_e_salvar():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    with open("ip_test.txt", "r") as f:
        ips = [linha.strip() for linha in f if linha.strip()]

    for ip in ips:
        print(f"🔍 Consultando IP: {ip}")
        relatorio = consultar_virustotal(ip)
        if relatorio:
            resultado = classificar(relatorio)
            classificacao = resultado["classificacao"]
            tipos = resultado["tipos_detectados"]

            
            text_pln = clean_text(json.dumps(relatorio))
            resultado_pln = classificar_pln(text_pln, MODELOS_PLN)

            print("Resultado LSTM:")
            print(f"\tClassificação: {classificacao}")
            print(f"\tTipos: {tipos}")

            print("\nResultado PLN:")
            print(f"Classificação: {resultado_pln}")

            ip_safe = ip.replace(".", "-")
            if tipos:
                tipos_str = "-".join(tipos).replace("/", "_")
                nome_arquivo = f"{classificacao}_{tipos_str}_{ip_safe}.txt"
            else:
                nome_arquivo = f"{classificacao}_{ip_safe}.txt"

            caminho = ASSETS_DIR / nome_arquivo

            with open(caminho, "w") as f:
                json.dump(relatorio, f, indent=2)

            print(f"\nSalvo como {caminho}\n")

        # Evitar rate limit
        time.sleep(15)

if __name__ == "__main__":
    coletar_e_salvar()