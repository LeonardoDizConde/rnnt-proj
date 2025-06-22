# TheHive Webhook + Cortex + PLN

Este projeto implementa um webhook Python integrado ao TheHive e Cortex para classificar observáveis de IP com base em análise automatizada por PLN e enriquecer alertas.

## Principais recursos:
- Análise automática com VirusTotal e AbuseIPDB via Cortex
- Classificação com LSTM
- Enriquecimento e comentários automáticos no TheHive
- Treinamento supervisionado com relatórios em .txt

## Como executar:
1. Ative o ambiente virtual
2. Execute o script `modelo_pln.py` para treinar
3. Rode `background_alert_processor_pln.py` para escutar alertas

## Autores
Jonas Aguiar
Keli Tauana
Leonardo Conde
