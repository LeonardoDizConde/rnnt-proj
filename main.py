import requests
import json
import time
from pathlib import Path
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configurações
VIRUSTOTAL_API_KEY = "b0ff5308c877a949a81bcf43913c0c52098fe915e88e0b9fab9ffd934d9ade65"
HEADERS = {"x-apikey": VIRUSTOTAL_API_KEY}
ASSETS_DIR = Path("./assets")
ASSETS_DIR.mkdir(exist_ok=True)

# Índice das classes - ajuste conforme seu treino
INDEX_TO_LABEL = ["limpo", "malicioso"]

# Carregar modelo treinado
model = load_model("models/lstm/best_model.keras")

# Carregar tokenizer treinado
with open("./models/lstm/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Função para limpar texto (adicione a sua lógica aqui)
def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ").replace("\r", " ")
    return text

# Consulta VirusTotal por IP
def consultar_virustotal(ip):
    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"[ERRO] IP {ip} - Status {response.status_code}")
        return None

# Classificar usando LSTM treinado
def classificar(relatorio):
    try:
        # Transformar JSON em string
        raw = json.dumps(relatorio)

        # Extrair tipos detectados (opcional)
        resultados = relatorio["data"]["attributes"].get("last_analysis_results", {})
        detectados = [dados["engine_name"] for dados in resultados.values() if dados["category"] == "malicious"]

        classifications_text = " ".join(detectados) if detectados else ""

        # Concatenar texto como no treino
        full_text = f"{raw} {classifications_text}"

        # Limpar texto
        texto_limpo = clean_text(full_text)

        # Transformar em sequência de tokens
        sequencia = tokenizer.texts_to_sequences([texto_limpo])

        # Padronizar tamanho da sequência (ajuste maxlen conforme treino)
        X_input = pad_sequences(sequencia, maxlen=200)

        # Fazer predição
        pred = model.predict(X_input)

        # Interpretar classe
        classe = int(np.argmax(pred, axis=1)[0])

        classificacao = INDEX_TO_LABEL[classe]

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

            print("Resultado:")
            print(f"\tClassificação: {classificacao}")
            print(f"\tTipos: {tipos}")
            # Criar nome de arquivo seguro
            ip_safe = ip.replace(".", "-")
            if tipos:
                tipos_str = "-".join(tipos)
                nome_arquivo = f"{classificacao}_{tipos_str}_{ip_safe}.txt"
            else:
                nome_arquivo = f"{classificacao}_{ip_safe}.txt"

            caminho = ASSETS_DIR / nome_arquivo

            # Salvar relatório JSON
            with open(caminho, "w") as f:
                json.dump(relatorio, f, indent=2)

            print(f"\nSalvo como {caminho}\n")

        # Evitar rate limit
        time.sleep(15)

if __name__ == "__main__":
    coletar_e_salvar()
