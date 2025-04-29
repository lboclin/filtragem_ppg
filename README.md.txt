# Filtragem Inteligente de Sinais PPG para Cães🐕

Este projeto implementa uma solução baseada em Inteligência Artificial (GANs) para filtragem de sinais PPG captados em cães.

## Estrutura do Projeto

- `data/` - Dados brutos, processados e ruidosos (não incluídos no repositório).
- `outputs/` - Gráficos gerados pela avaliação (não incluídos no repositório).
- `scripts/` - Scripts de processamento, adição de ruído, treino e avaliação.

## Fluxo de Execução

1. `load_and_download_ppg.py` - Carrega sinais brutos.
2. `segment_and_save.py` - Segmenta sinais em janelas de 30s.
3. `add_noise.py` - Adiciona ruído branco aos sinais.
4. `gan_generator.py` e `gan_discriminator.py` - Definem a arquitetura da GAN.
5. `train_gan.py` - Treina a GAN.
6. `evaluate_generator.py` - Avalia a GAN em um sinal.
7. `batch_evaluate_and_save.py` - Avalia todos os sinais e salva os gráficos.

## Observação

Este projeto foi validado com ruído branco simulado. Próximos passos incluem coleta de dados reais de cães militares e ajuste do treinamento para ambientes reais.

---
