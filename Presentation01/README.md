# Apresentação 01 — Artigo em ML aplicado à geofísica

Este diretório reúne os materiais da tarefa **Apresentação de artigo em ML aplicado à geofísica**, da disciplina **PPGEEC2327 — Tópicos Especiais em Processamento Inteligente da Informação**, turma T01, período 2026.2, do PPgEEC/UFRN.

O trabalho foi preparado pela dupla **Lucas Torres Marques e Rosélia Nascimento**, com foco no uso de aprendizado profundo para atenuar ruído de swell em dados sísmicos marinhos.

## Enunciado da tarefa

A atividade consiste em apresentar, **em dupla e em sala de aula**, um artigo científico publicado recentemente sobre uma aplicação real de Machine Learning à geofísica. A apresentação deve:

- abordar a teoria e os resultados do artigo;
- explicar a técnica de Machine Learning utilizada;
- identificar o problema geofísico estudado;
- respeitar o limite de **15 minutos**.

Devem ser enviados os PDFs do artigo (ou relatório) e dos slides. Os nomes dos integrantes da dupla devem ser enviados por e-mail ao docente.

## Artigo selecionado

**BARROS, Pablo M. et al.** *An open benchmark dataset of synthetic seismic data and real swell noise for evaluating deep learning denoising models*. Artificial Intelligence in Geosciences, v. 7, artigo 100219, 2026.

- **DOI:** [10.1016/j.aiig.2026.100219](https://doi.org/10.1016/j.aiig.2026.100219).
- **Instituições dos autores:** COPPE/UFRJ e CENPES/Petrobras.
- **Tema:** construção de um benchmark aberto para avaliar modelos de aprendizado profundo na remoção de ruído sísmico, combinando sinais sintéticos com ruído real de swell.

## Problema geofísico e técnicas de ML

O ruído de swell está associado às condições do mar e ao movimento dos cabos receptores (*streamers*) durante a aquisição sísmica marinha. Ele contamina os registros e dificulta a interpretação dos refletores. Como suas componentes podem se sobrepor às do sinal de interesse, a filtragem precisa reduzir o ruído preservando a informação geológica.

O estudo avalia redes neurais convolucionais para remoção de ruído (*denoising*), com os modelos **FCNN-5, SRGEN-4 e SRGEN-16**, além de um filtro passa-alta como referência clássica. O benchmark combina quatro conjuntos de dados sísmicos sintéticos, dois arquivos de ruído real e quatro níveis de relação sinal-ruído, organizados em **32 configurações de treinamento e teste** com separação entre os dados utilizados em cada etapa.

Essa combinação permite avaliar a remoção de ruído real usando um sinal limpo conhecido como referência. A apresentação também discute a métrica **SNR²**, proposta no artigo para complementar a avaliação da preservação do sinal.

## Conteúdo da apresentação

Os slides abordam:

1. contexto da aquisição sísmica marinha e características do ruído de swell;
2. objetivos do artigo e construção do conjunto de dados;
3. fundamentos das arquiteturas FCNN e SRGEN;
4. protocolo experimental e métricas de avaliação;
5. resultados quantitativos e inspeção visual das reconstruções;
6. contribuições, limitações e conclusões.

A discussão destaca que o desempenho depende das características dos dados e do nível de ruído. A preservação dos refletores é tão relevante quanto a redução do ruído: uma saída visualmente mais limpa pode conter perda de sinal útil. Entre as limitações discutidas estão o uso de sinais limpos sintéticos, o foco em ruído de swell e os desafios de generalização.

## Arquivos e consulta

| Arquivo | Conteúdo |
|---|---|
| [swell_noise_ufrj.pdf](./swell_noise_ufrj.pdf) | Texto completo do artigo selecionado. |
| [Apresentação de artigo em ML aplicado à geofísica (Lucas e Rosélia).pdf](<./Apresentação de artigo em ML aplicado à geofísica (Lucas e Rosélia).pdf>) | Slides preparados pela dupla para a apresentação em sala. |
| [README.md](./README.md) | Descrição da tarefa, do estudo e dos materiais. |

Os materiais podem ser consultados em qualquer leitor de PDF, sem instalação de ambiente Python ou execução de código. Os experimentos e resultados discutidos são os apresentados no artigo.

## Dupla

- [Lucas Torres Marques](https://github.com/Lucastmarques)
- [Rosélia Nascimento](https://github.com/roseliasilva)
