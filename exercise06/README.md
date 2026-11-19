# Exercício 06 — Atenuação de ruído sísmico com CSC

Conversão para Python do laboratório [**Convolutional Sparse Coding for Noise Attenuation of Seismic Data**](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.Sparse/Chapter.Sparse2/lab.html), de Zhaolun Liu. O [notebook](exercise06.ipynb) reúne o roteiro, a execução, as figuras e as análises em Markdown.

O exercício investiga como o tamanho e a quantidade dos filtros e o peso da penalização L1 afetam a reconstrução de dados sísmicos. Também compara os filtros CSC com os da primeira camada de uma CNN. O treinamento aprende filtros e mapas de coeficientes por ADMM; na remoção de ruído, os filtros ficam fixos e apenas os coeficientes são estimados.

## Arquivos

```text
exercise06/
├── README.md
├── exercise06.ipynb
├── conda.yaml
├── cnn_reference.npz
├── src/seismic_csc/
│   ├── __init__.py
│   ├── csc.py             # pré-processamento, operadores, ADMM e SNR
│   ├── data.py            # download e leitura dos dados oficiais
│   ├── experiments.py     # experimentos, métricas e exportação
│   └── plotting.py        # figuras
└── original/              # materiais MATLAB, cache e resultados
```

Os materiais de origem são descritos em [SOURCES.json](original/SOURCES.json). O pacote `original/csc_code.zip` é incluído no versionamento para fornecer os MAT necessários à execução sem um novo download. Os fontes extraídos permanecem locais, conforme o `.gitignore`. O notebook lê os MAT diretamente do ZIP, sem depender da extração MATLAB; se o pacote estiver ausente, tenta baixá-lo do site oficial.

## Preparar e executar

A partir de `exercise06/`, crie o ambiente e registre o kernel:

```bash
conda env create --file conda.yaml
conda activate exercise06
python -m ipykernel install --user --name exercise06 --display-name "Python (exercise06)"
jupyter lab exercise06.ipynb
```

Selecione **Python (exercise06)** e execute **Run All**. Para atualizar um ambiente existente, use `conda env update --file conda.yaml --prune`.

A execução usa CPU e precisão dupla, com NumPy, SciPy, Pandas e Matplotlib. MATLAB e PyTorch não são necessários. A configuração fica na primeira célula de código; as saídas são atualizadas em `original/results/main/`. As tabelas e figuras são recalculadas, enquanto as anotações quantitativas em Markdown descrevem a execução de referência e devem ser revistas se os parâmetros mudarem.

O pacote é verificado antes da leitura pelo SHA-256:

```text
fe5bd09472bd0f2d334450be19fe1f8ce56864b536e3a5d26eaa8fe1fd3fd44f
```

Se o download automático falhar, salve [csc_code.zip](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.Sparse/Chapter.Sparse2/csc_code.zip) em `original/csc_code.zip`. Um pacote com hash diferente é recusado. Os arquivos MATLAB fornecidos são preservados.

## Dados e experimentos

`b.mat` contém cinco registros de tiro comum (CSGs), em uma matriz `181 × 100 × 5`. O treinamento usa as primeiras 100 amostras do CSG 1, e a avaliação usa o mesmo recorte do CSG 2. Os demais CSGs aparecem apenas na inspeção. Como o MAT não fornece coordenadas físicas, os gráficos usam índices de amostra e traço.

| Parâmetro | Referência |
|---|---|
| Pré-processamento | Contraste local gaussiano `21 × 21`, sigma `4,773`, seguido de remoção da média |
| Dicionário | Três filtros de `21 × 11` |
| Penalizações | L1 no treino `1`; L1 na reconstrução `0,5`; fidelidade ao dado `1` |
| Ruído | Gaussiano com desvio padrão `0,3`, adicionado após o pré-processamento |
| Sementes | Treino `2026`; ruído `2027` |
| Iterações | Até 80 externas no treino, com 10 atualizações por variável; até 80 na reconstrução |
| Tolerância | Mudança relativa `1e-3` |

As séries variam um fator por vez: tamanhos `11 × 7`, `21 × 11` e `31 × 15`; quantidades `K = 1, 3, 6`; e penalizações de reconstrução `beta = 0,1; 0,25; 0,5; 1; 2; 5`. A mesma imagem ruidosa é usada em todas as comparações. O dicionário central é reutilizado, totalizando cinco treinamentos e onze reconstruções, incluindo uma com os filtros MATLAB fornecidos.

## Pontos importantes da conversão

| MATLAB | Python em `src/seismic_csc/` |
|---|---|
| `training_phase.m` e `denoise.m` | Notebook e `experiments.py` |
| `admm_learn_conv2D_weighted_test.m` | `csc.py`: `learn_dictionary` |
| `admm_solve_conv2D_weighted_sparse_reconstruction.m` | `csc.py`: `sparse_reconstruct` |
| `local_contrast_normalization.m` | `csc.py`: `local_contrast_normalization` |
| `snr_test.m` | `csc.py`: `snr_db` |
| Rotinas gráficas | `plotting.py`, com Matplotlib |

- **Convolução e bordas:** as FFTs atuam nos eixos espaciais `(0, 1)`. Os filtros são preenchidos com zeros e centralizados como em `psf2otf`; as métricas usam o recorte observado. Com filtros `(h, w)`, os coeficientes têm dimensões espaciais `(H+h-1, W+w-1)`. São aceitos suportes positivos e ímpares.
- **Normalização:** a reflexão não repete o pixel da borda. O piso do desvio padrão usa o elemento central inferior para uma quantidade par de valores, preservando o MATLAB em vez da interpolação de `numpy.median`. Variâncias negativas por arredondamento são truncadas em zero e denominadores nulos recebem epsilon.
- **ADMM:** preservam-se a limiarização suave, a projeção dos filtros e as penalidades internas baseadas em `gamma = 60*beta/max(x)`. A atualização de filtros usa Woodbury e a de coeficientes usa Sherman–Morrison. O treino restaura o estado anterior quando nenhuma direção melhora o critério original; o motivo de parada e as iterações são registrados.
- **Filtros retornados:** a projeção atua em uma variável auxiliar. O iterado retornado pode violar ligeiramente a norma unitária, como nos filtros MATLAB fornecidos, cujas normas são aproximadamente `1,0104`, `1,0104` e `1,0205`. A saída não é renormalizada automaticamente. O treinamento usa D estendido; a reconstrução posterior usa os filtros compactos.
- **Reprodutibilidade:** as sementes de treino e ruído são separadas. Cada reconstrução começa com coeficientes e variáveis duais nulos. Versões, configurações e hashes são exportados. Sementes iguais em linguagens diferentes não garantem estados iniciais iguais.

Durante o desenvolvimento, dez verificações numéricas cobriram convolução direta versus FFT, sistemas complexos versus soluções densas, normalização, projeção, máscara e repetibilidade. O notebook foi executado integralmente. MATLAB não foi executado nessa validação; a comparação com os filtros fornecidos não equivale a comparar duas execuções com inicialização idêntica.

## Resultados de referência

A entrada ruidosa apresentou **7,0755 dB** de SNR.

| Configuração | SNR de saída (dB) |
|---|---:|
| Referência Python: `21 × 11`, K = 3, beta = 0,5 | 10,0869 |
| Filtros MATLAB fornecidos, mesmo ruído e beta | 15,1423 |
| Filtros `11 × 7`, K = 3 | 14,2925 |
| Filtros `31 × 15`, K = 3 | 11,7060 |
| Filtros `21 × 11`, K = 1 | 3,5355 |
| Filtros `21 × 11`, K = 6 | 12,9887 |
| Dicionário de referência, beta = 0,25 | 11,7815 |
| Dicionário de referência, beta = 5 | 0,3496 |

O menor suporte foi o melhor na série de tamanhos, seis filtros superaram três na série de K e beta `0,25` foi o melhor valor da série L1. Penalizações altas reduziram a atividade dos coeficientes, mas também retiraram eventos sísmicos, visíveis nos residuais.

Esses resultados descrevem um CSG de avaliação, uma realização de ruído e um orçamento fixo de iterações. Selecionar parâmetros pela mesma SNR não constitui teste independente de generalização. A avaliação ocorre na escala normalizada, sem recuperação de amplitudes físicas. Os filtros fornecidos e os novos representam soluções diferentes de um problema não convexo; sua diferença de SNR não isola o efeito da linguagem.

## Resultados exportados e comparação CNN

Em `original/results/main/`, `summary.csv` reúne as métricas de reconstrução e `training_summary.csv` resume os treinamentos. `settings.json` registra parâmetros, versões e hashes. As pastas `training/` e `denoising/` contêm estados NPZ, históricos CSV e configurações; `figures/` contém as figuras PNG.

A SNR é calculada por `10 log10(soma(x²) / soma((estimativa − x)²))`. A fração ativa usa `|z| > 10⁻³ max(|z|)` no domínio estendido, com limiar absoluto exportado. Ela não representa uma contagem de zeros de uma solução totalmente convergida. Objetivos com betas diferentes também não são diretamente comparáveis como medida de qualidade.

`cnn_reference.npz` contém os seis filtros `5 × 5` da primeira camada do modelo binário de 100 épocas do [exercício 04](../exercise04/README.md), com configuração e SHA-256 do checkpoint no campo `metadata_json`. O notebook lê esse arquivo com NumPy. A comparação é visual e conceitual: ambos aprendem padrões locais compartilhados, mas o CSC reconstrói sinais com coeficientes esparsos, enquanto a CNN foi treinada para classificação supervisionada de fraturas em rocha.

## Fontes

- [Laboratório — Zhaolun Liu](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.Sparse/Chapter.Sparse2/lab.html).
- [Resumo teórico](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.Sparse/Chapter.Sparse2/CSCdenoising.pdf) e [slides](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.Sparse/Chapter.Sparse2/csc.pptx).
- F. Heide, W. Heidrich e G. Wetzstein, *Fast and Flexible Convolutional Sparse Coding*, CVPR 2015: autores do núcleo indicado nos cabeçalhos MATLAB, adaptado por Zhaolun Liu.
