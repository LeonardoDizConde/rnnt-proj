import requests
import time
import os
import json
from pathlib import Path

# Configurações
VIRUSTOTAL_API_KEY = "b0ff5308c877a949a81bcf43913c0c52098fe915e88e0b9fab9ffd934d9ade65"
HEADERS = {"x-apikey": VIRUSTOTAL_API_KEY}
ASSETS_DIR = Path("./new_assets")
ASSETS_DIR.mkdir(exist_ok=True)

# Classificação com base nas regras
def classificar(relatorio):
    attr = relatorio.get("data", {}).get("attributes", {})
    stats = attr.get("last_analysis_stats", {})
    reputation = attr.get("reputation", 0)
    tags = attr.get("tags", [])
    votos = attr.get("total_votes", {}).get("malicious", 0)

    mal = stats.get("malicious", 0)
    susp = stats.get("suspicious", 0)

    # Determinar classificação geral
    if mal >= 10 or reputation <= -50 or votos >= 5 or "tor" in tags:
        classificacao = "malicioso"
    elif 1 <= mal < 10 or susp >= 1 or reputation < 0:
        classificacao = "suspeito"
    else:
        classificacao = "benigno"

    # Tipos de ameaça conhecidos a partir das tags
    tipos_possiveis = [
        "scam", "spam", "malware", "phishing", "botnet", "command-and-control",
        "cryptomining", "hijacked", "ddos", "proxy", "anonymization-service",
        "ransomware", "backdoor"
    ]

    # Interseção das tags com os tipos conhecidos
    tipos_detectados = [t for t in tags if t in tipos_possiveis]

    return {
        "classificacao": classificacao,
        "tipos_detectados": tipos_detectados
    }

# Consulta VT por IP
def consultar_virustotal(ip):
    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"[ERRO] IP {ip} - Status {response.status_code}")
        return None

# Loop principal
def coletar_e_salvar():
    # Garantir que o diretório exista
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    with open("ip_mal.txt", "r") as f:
        ips = [linha.strip() for linha in f if linha.strip()]

    ips_invertidos = ips[::-1]

    for ip in ips_invertidos:
        print(f"🔍 Consultando IP: {ip}")
        relatorio = consultar_virustotal(ip)
        if relatorio:
            resultado = classificar(relatorio)
            classificacao = resultado["classificacao"]
            tipos = resultado["tipos_detectados"]

            # Criar nome de arquivo seguro
            ip_safe = ip.replace(".", "-")
            if tipos:
                tipos_str = "-".join(tipos)
                nome_arquivo = f"{classificacao}_{tipos_str}_{ip_safe}.txt"
            else:
                nome_arquivo = f"{classificacao}_{ip_safe}.txt"

            caminho = ASSETS_DIR / nome_arquivo

            # Salvar o relatório como JSON
            with open(caminho, "w") as f:
                json.dump(relatorio, f, indent=2)

            print(f"✅ Salvo como {caminho}")

        time.sleep(15)  # Evita ultrapassar rate limit

if __name__ == "__main__":
    coletar_e_salvar()
