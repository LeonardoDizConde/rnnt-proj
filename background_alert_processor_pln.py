from flask import Flask, request, jsonify
import requests
import logging
from openai import OpenAI
import time
import os
from datetime import datetime
import joblib
import json
import spacy
import nltk
import string
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# Configurações
hive_url = os.getenv("THEHIVE_URL")  # Substitua pelo URL do seu servidor The Hive
hive_api_key = os.getenv("THEHIVE_API_KEY")  # Substitua pela sua chave API do The Hive

headers = {
    'Authorization': f'Bearer {hive_api_key}',
    'Content-Type': 'application/json'
}

# Carrega modelos treinados
MODELOS_PLN = {
    "bow_nb": joblib.load("/home/thehive/thehive-webhook/models/bow_nb.joblib"),
    "tfidf_svc": joblib.load("/home/thehive/thehive-webhook/models/tfidf_svc.joblib")
}

# Funções auxiliares
nlp = spacy.load("en_core_web_sm")
nltk.download("stopwords")
stopwords = nltk.corpus.stopwords.words("english")

def clean_text(text):
    tokens = nlp(text)
    tokens = [str(t).lower() for t in tokens if str(t) not in string.punctuation]
    tokens = [str(t) for t in tokens if str(t) not in stopwords]
    return " ".join(tokens)

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


def analyze_with_pln_and_post_comment(alert_id, analyzer_results):
    all_comments = []
    
    if analyzer_results:
        for result in analyzer_results:
            raw_json = result
            text = clean_text(json.dumps(raw_json))
            label = ensemble_predict(text, MODELOS_PLN)
            comment = generate_comment(label, raw_json)
            all_comments.append(f"{comment}")

    combined_comments = "\n\n".join(all_comments)

    if combined_comments:
        post_comment_to_case(alert_id, combined_comments)
        
def generate_comment(label: str, raw_json: dict) -> str:
    attr = raw_json["full"]["attributes"]
    ip = raw_json.get("full", {}).get("id", "N/A")
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
            f"Indicador classificado como malicioso com {mal} detecções confirmadas por mecanismos de análise. "
            f"O IP {ip} está associado ao ASN '{as_owner}' localizado em {country}. "
            f"O endereço tem reputação negativa ({reputation}) e apresenta as seguintes tags: {tags}. "
            f"Recomenda-se o bloqueio do IP e investigação sobre possíveis conexões anteriores."
        )
    elif label == "suspeito":
        return (
            f"Indicador classificado como suspeito com {susp} alertas levantados por ferramentas de reputação. "
            f"O IP {ip}, pertencente a '{as_owner}' ({country}), exige monitoramento contínuo."
        )
    else:
        return (
            f"Indicador classificado como benigno com {harmless} mecanismos classificando como inofensivo. "
            f"O IP {ip} pertence a '{as_owner}' e não apresenta sinais atuais de risco."
        )
        
def ensemble_predict(text, models):
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


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    logging.info(f"Received webhook: {data}")
    
    observable = data.get('objectId')  # Obtém o ID do observável
    
    
    # Se vier uma lista, extrai o primeiro elemento
    if isinstance(observable, list):
        observable = observable[0] if observable else None
    
    if not isinstance(observable, str):
        logging.error(f"observable inválido ou ausente: {observable}")
        return jsonify({"status": "invalid observable"}), 400
    
    case_id = data.get('rootId')
    alert_id = data.get('object', {}).get('alert', {}).get('_id')
    
    alert_time  = data.get('object', {}).get('context', {}).get('startDate')
        
    analyzers_id = list_analyzers()
    severity = data.get('context', {}).get('severityLabel')
    job_ids = run_analyzers(observable,analyzers_id)
    
    if severity in ('MEDIUM','HIGH', 'CRITICAL','LOW'):
        logging.info(f"Severity found: {severity}")
        analyzer_results  = wait_for_jobs_and_get_results(job_ids)
        
        analyze_with_pln_and_post_comment(alert_id, analyzer_results)
        
        mal_report = verify_malicious(analyzer_results)
        
        if mal_report:
            logging.info("Malicious report found. Creating case.")
            create_case_from_alert(alert_id)  # Cria um caso se houver maliciosidade
        elif severity in ('MEDIUM','HIGH', 'CRITICAL','LOW'):
            create_case_from_alert(alert_id)  # Cria um caso se houver maliciosidade
        else:
            logging.info("No case will be created.")
        
    return jsonify({"status": "success"}), 200

def is_out_of_business_hours(alert_time):
    """
    Verifica se o alerta foi criado fora do horário comercial (das 8h às 18h)
    ou em finais de semana ou feriados.
    """
    # Converte o timestamp em milissegundos para uma data legível
    alert_datetime = datetime.utcfromtimestamp(alert_time / 1000)  # Convertendo para datetime UTC
    current_time = alert_datetime.time()
    
    # Verifica se é fim de semana
    if alert_datetime.weekday() >= 5:  # 5 é sábado, 6 é domingo
        return True

    # Verifica se está fora do horário comercial (8h às 18h)
    business_start = time(8, 0)
    business_end = time(18, 0)
    if not (business_start <= current_time <= business_end):
        return True

    return False
 
def list_analyzers():
    """Lista todos os analisadores disponíveis no Cortex"""
    url = f"{hive_url}/api/connector/cortex/analyzer"
    response = requests.get(url, headers=headers)
    logging.debug(f"list_analyzers response: {response.status_code} - {response.text}")
    if response.status_code == 200:
        return [analyzer['id'] for analyzer in response.json()]
    else:
        logging.error(f"Failed to list analyzers: {response.status_code} - {response.text}")
        return []

def run_analyzers(observable, analyzers_id):
    """Executa todos os analisadores disponíveis nos observables"""
    job_ids = []
    for analyzer in analyzers_id:
        url = f"{hive_url}/api/connector/cortex/job"
        data = {
            "analyzerId": analyzer,
            "cortexId": 'cortex0',
            "artifactId":observable
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 201:
            logging.info(f"Successfully created job for analyzer {analyzer} on observable {observable}")
            job_ids.append(response.json()['_id'])  # Adiciona o job_id à lista
        else:
            logging.error(f"Failed to run: {response.status_code} - {response.text}")
    return job_ids if job_ids else None

def wait_for_jobs_and_get_results(job_ids):
    """Espera até que todos os jobs sejam concluídos e retorna os resultados, limitados a 60.000 tokens"""
    results = []
    total_tokens = 0  # Contador de tokens

    for job_id in job_ids:
        while True:
            url = f"{hive_url}/api/connector/cortex/job/{job_id}"
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                job = response.json()
                if job['status'] == 'Success':
                    results.append(job['report'])

                    # Estima o número de tokens (aproximadamente 1 token a cada 4 caracteres)
                    report_tokens = len(results) // 4

                    # Verifica se adicionar o próximo relatório excede o limite de 60.000 tokens
                    if total_tokens + report_tokens > 60000:
                        logging.info(f"Token limit reached with {total_tokens} tokens, stopping addition of further reports.")
                        return results  # Retorna os resultados acumulados até o momento

                    total_tokens += report_tokens
                    
                    break

                elif job['status'] in ['Failure', 'Error']:
                    logging.error(f"Job {job_id} failed with status: {job['status']}")
                    break
                else:
                    logging.info(f"Job {job_id} still in progress, waiting...")
                    time.sleep(5)  # Espera 5 segundos antes de verificar novamente
            else:
                logging.error(f"Failed to get job status for {job_id}: {response.status_code} - {response.text}")
                break

    return results
        
def analyze_with_gpt_and_post_comment(alert_id, analyzer_results):
    all_comments = []  # Lista para armazenar todas as análises
    
    """Envia os resultados dos analyzers para GPT-4o-mini para análise contextual e posta o resultado nos comentários"""
    if analyzer_results:
        prompt = f"""
            Baseado nos seguintes relatórios dos analisadores: {analyzer_results}, providencie uma análise contextual de segurança cibernética. A resposta deve seguir o formato abaixo e deve ser estruturada para fácil inclusão em comentários de casos no The Hive. Inclua conclusões e recomendações específicas.

            **Instruções Adicionais**:
            - Após a leitura de todos os relatórios, realize a análise e escreva uma única resposta de segurança.
            - Evite informações redundantes e mantenha o foco em análises objetivas e úteis.
        """
        try:
            comment = send_to_gpt(prompt)
            all_comments.append(comment)  # Adiciona cada comentário à lista
        except Exception as e:
            logging.error(f"Error while analyzing with GPT: {e}")
            time.sleep(60)
            
    # Concatena todas as análises em uma única string
    combined_comments = "\n\n".join(all_comments)
    
    # Atualiza a descrição do alerta com todas as análises concatenadas
    if combined_comments:
        post_comment_to_case(alert_id, combined_comments)  # Atualiza a descrição com todos os observáveis            

def verify_malicious(analyzer_results):
    # Inicialmente assumimos que não há maliciosidade
    malicious_found = False  
    
    if analyzer_results:
        for report in analyzer_results:
            if 'full' in report:
                if 'attributes' in report['full']:
                    attributes = report['full']['attributes']
                    if 'last_analysis_stats' in attributes and attributes['last_analysis_stats'].get('malicious', 0) > 0:
                        malicious_found = True
                        
                if 'values' in report['full']:
                    for value in report['full']['values']:
                        if 'data' in value and 'abuseConfidenceScore' in value['data']:
                            if value['data']['abuseConfidenceScore'] > 0:
                                malicious_found = True
    
    return malicious_found
       
def send_to_gpt(prompt):
    """Envia um prompt para o GPT-4o-mini e retorna a resposta"""
    gpt_api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=gpt_api_key)
    prompt_system = f"""
            Você é um analista de segurança cibernética altamente qualificado que trabalha em um SOC (Security Operation Center). Sua tarefa é realizar análises detalhadas e contextualizadas de segurança cibernética para relatórios gerados por analisadores de segurança, como AbuseIPDB, VirusTotal, entre outros.

            **Instruções para a Análise**:
            - Cada análise deve ser clara, objetiva e seguir uma estrutura lógica.
            - Você deve descrever o contexto geral do observável, incluindo suas características principais e relevância.
            - Avalie a reputação e os resultados das análises realizadas pelos diferentes analisadores e forneça um resumo claro desses resultados.
            - Quando aplicável, inclua detalhes sobre certificações e assinaturas digitais associadas ao observável.
            - Descreva o comportamento observado, como comunicações de rede ou atividades que possam ser consideradas suspeitas.
            - Se o observável interagir com outros IPs ou domínios, descreva essas interações e discuta suas implicações para a segurança.
            - Para cada observável, forneça uma conclusão e recomendações específicas, incluindo a classificação do risco e as ações recomendadas, como bloqueio ou monitoramento.
            
            **Outras Diretrizes**:
            - Concentre-se apenas nos observáveis ou relatórios que contêm dados relevantes.
            - Estruture cada análise de forma clara para fácil inclusão em comentários de casos no The Hive.
            - Evite redundâncias e mantenha a análise objetiva e focada no essencial.
            
            Seu objetivo é garantir que cada relatório seja analisado de forma crítica e precisa, resultando em recomendações acionáveis que ajudam a mitigar riscos de segurança cibernética.
            """
    try:
        data = client.chat.completions.create(
            messages = [
                {
                    "role":"system",
                    "content":prompt_system
                },
                {
                "role":"user",
                "content":prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000,
            model="gpt-4o-mini")
        comment = data.choices[0].message.content
        return comment.strip()
    except Exception as e:
        print(f"Error analyzing results with GPT: {e}")
        return "Análise não disponível"

def post_comment_to_case(alert_id, comment):
    """Adiciona a análise do GPT na descrição do alerta no The Hive"""
    # Obter o alerta atual para concatenar a descrição
    url_get = f"{hive_url}/api/v1/alert/{alert_id}"
    response_get = requests.get(url_get, headers=headers)
    
    if response_get.status_code == 200:
        alert_data = response_get.json()
        existing_description = alert_data.get("description", "")
        
        # Adiciona o novo comentário à descrição existente
        new_description = f"{existing_description}\n\nAnálise PLN:\n{comment}"
        
        # Atualiza a descrição do alerta
        data = {
            "description": new_description
        }
        response_patch = requests.patch(url_get, headers=headers, json=data)
        
        if response_patch.status_code == 200:
            logging.info(f"Successfully updated description for alert {alert_id}")
        else:
            logging.error(f"Failed to update description for alert {alert_id}: {response_patch.status_code} - {response_patch.text}")
    else:
        logging.error(f"Failed to retrieve alert data for alert ID {alert_id}: {response_get.status_code} - {response_get.text}")

def create_case_from_alert(alert_id):
    """
    Cria um caso no TheHive baseado no alerta original, herdando todas as suas informações.
    """  
    # Obtém as informações do alerta original para herdar no caso
    case_url = f"{hive_url}/api/v1/alert/{alert_id}/case"
    case_response = requests.post(case_url, headers=headers)
    
    if case_response.status_code == 201:
            case_id = case_response.json()['_id']
            logging.info(f"Successfully created case {case_id} from alert {alert_id}")  
    else:
        logging.error(f"Failed to retrieve alert data for alert ID {alert_id}: {case_response.status_code} - {case_response.text}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
