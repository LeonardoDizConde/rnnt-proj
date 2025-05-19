import requests
import time
import os
import json
from pathlib import Path

# Configurações
VIRUSTOTAL_API_KEY = "b0ff5308c877a949a81bcf43913c0c52098fe915e88e0b9fab9ffd934d9ade65"
HEADERS = {"x-apikey": VIRUSTOTAL_API_KEY}
ASSETS_DIR = Path("./assets")
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

    if mal >= 10 or reputation <= -50 or votos >= 5 or "tor" in tags:
        return "malicioso"
    elif 1 <= mal < 10 or susp >= 1 or reputation < 0:
        return "suspeito"
    else:
        return "benigno"

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
    with open("ip_mal.txt", "r") as f:
        ips = [linha.strip() for linha in f if linha.strip()]

    for ip in ips:
        print(f"🔍 Consultando IP: {ip}")
        relatorio = consultar_virustotal(ip)
        if relatorio:
            classe = classificar(relatorio)
            caminho = ASSETS_DIR / f"{classe}_{ip.replace('.', '-')}.txt"
            with open(caminho, "w") as f:
                json.dump(relatorio, f, indent=2)
            print(f"✅ Salvo como {caminho}")
        time.sleep(15)  # Evita ultrapassar o rate limit gratuito (4/min)

if __name__ == "__main__":
    coletar_e_salvar()
