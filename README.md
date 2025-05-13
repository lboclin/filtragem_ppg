
## Etapas Realizadas

1. **Download de Dados**
   - Dataset: [`pulse-transit-time-ppg`](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/)
   - Sinais PPG extraídos do canal `pleth_*`.

2. **Pré-processamento**
   - Segmentação em janelas de 30 segundos
   - Normalização por Z-score

3. **Geração de Ruído**
   - `noisy/`: ruído branco adicionado artificialmente
   - `noisy_realistic/`: ruído extraído de trechos reais de sensores

4. **Treinamento da GAN**
   - Gerador baseado em CNN
   - Discriminador baseado em Fully Connected
   - Treinamento supervisionado para reconstruir sinal limpo

5. **Avaliação**
   - Geração de gráficos comparativos (PPG limpo vs ruidoso)
   - Salvos em `outputs/`

## Scripts principais

| Script | Função |
|--------|--------|
| `load_and_download_ppg.py` | Baixa e extrai registros `.hea` e `.dat` |
| `segment_ppg_signal.py` | Corta o PPG em segmentos de 30s |
| `normalize_ppg.py` | Aplica normalização Z-score |
| `generate_noisy_segments.py` | Adiciona ruído branco aos segmentos |
| `add_realistic_noise.py` | Injeta ruído realista baseado em sinais reais |
| `train_gan.py` | Treina a GAN com dados ruidosos |
| `save_generator.py` | Salva o modelo do gerador |
| `evaluate_generator.py` | Gera comparação visual de saída da GAN |
| `plot_ppg_compare.py` | Plota e salva os gráficos de comparação |

## Exemplo de Gráfico

Cada gráfico mostra:
- Linha azul: PPG ruidoso (com artefatos)
- Linha laranja: PPG filtrado pela GAN

## ✅ Resultados

- A GAN treinada com ruído realista apresenta suavização consistente do sinal, mantendo os picos principais
- Arquitetura adaptável a novos datasets (ex: CIRCor Heart Sound)
- Projeto pronto para ser validado com **dados reais de sensores em cães**
