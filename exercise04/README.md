# Exercício 04 — Identificação de fraturas em rocha com CNN

Este diretório contém a conversão para Python/PyTorch do laboratório [**Identifying Rock Crack by Convolutional Neural Network**](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.CNN/Chapter.CNNCrack/lab.html), de Shihang Feng, associado à seção 10.5 e ao apêndice 10.18 de *Machine Learning Methods in Geoscience*.

A execução principal fica no notebook. As rotinas do pacote MATLAB `CNN_lab` foram adaptadas em um módulo Python. O notebook reúne a execução, as saídas e uma interpretação quantitativa e visual após cada experimento. A conclusão sintetiza as quatro questões e compara os problemas binário e multiclasse, incluindo a reunião das classes de fratura para comparar a detecção. Uma única execução contempla a classificação binária e a multiclasse a partir do mesmo picking manual.

## Arquivos

- `exercise04.ipynb`: configuração, roteiro, execução dos experimentos, figuras e coleta;
- `crack_cnn.py`: leitura dos dados, extração de patches, CNN, treinamento, métricas e exportação;
- `annotation_picker.py`: anotador Plotly para criar somente CSVs ausentes, sem sobrescrever arquivos existentes;
- `annotations_big.csv` e `annotations_small.csv`: seleções manuais de origem, reutilizadas sem alteração;
- `conda.yaml`: ambiente Conda próprio;
- `LICENSE-CNN_lab`: licença do código CNN original, de Rasmus Berg Palm;
- `.cache/CNN_lab.zip`: cache local criado no primeiro download, ignorado pelo Git;
- `results/`: arquivos gerados por execução, ignorados pelo Git.

Não é necessário instalar MATLAB, extrair o ZIP ou copiar dados para o repositório. O download oficial tem aproximadamente 1 MB. Seu SHA-256 é verificado antes da leitura:

```text
fd4279b568b1b3261d3c999e884fa70525608acf792815c5388fb4c19fdb6595
```

Se o diretório original `CNN_lab/` estiver presente, ele pode ser mantido para consulta; a execução Python não depende dele, e esses fontes ficam ignorados pelo Git, como os anexos originais do exercise02.

## 1. Criar o ambiente

Com Conda, Miniconda ou Mambaforge instalado:

```bash
cd exercise04
conda env create --file conda.yaml
conda activate exercise04
```

Para sincronizar um ambiente existente:

```bash
conda env update --file conda.yaml --prune
```

O YAML instala `torch==2.13.0+rocm7.2` pelo índice oficial do PyTorch para GPUs AMD em Linux/WSL, seguindo a configuração do exercise01. O driver e o runtime ROCm compatíveis devem estar instalados no sistema. A opção `DEVICE = "auto"` seleciona a GPU quando disponível; em GPUs AMD, a interface também se chama `torch.cuda`, enquanto `torch.version.hip` identifica a distribuição ROCm.

Se o ambiente foi criado com a configuração anterior para CPU, substitua o PyTorch com o ambiente `exercise04` ativo:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/rocm7.2 "torch==2.13.0+rocm7.2"
```

Depois da instalação, reinicie o kernel do notebook, selecione **Python (exercise04)** e execute novamente a primeira célula. Alterar apenas `DEVICE` não habilita uma GPU em uma distribuição CPU do PyTorch. Para usar outra plataforma, consulte o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/).

No WSL, o registro automático do profiler ROCm pode abortar a inicialização com `Found 0 rocprofiler agents`. O ambiente define `ROCPROFILER_REGISTER_ENABLED=0` para desabilitar esse registro. A primeira célula do notebook também aplica o ajuste antes de importar PyTorch quando identifica o WSL, inclusive se o kernel for iniciado diretamente pelo VS Code. Para atualizar um ambiente existente pelo terminal:

```bash
conda env config vars set --name exercise04 ROCPROFILER_REGISTER_ENABLED=0
```

Na verificação, `torch.version.hip` deve indicar ROCm, `torch.cuda.is_available()` deve retornar `True`, e `torch.cuda.get_device_name(0)` deve identificar a placa. Depois de uma falha do profiler ou da troca do pacote PyTorch, é necessário reiniciar o kernel.

## 2. Registrar o kernel

```bash
python -m ipykernel install --user \
  --name exercise04 \
  --display-name "Python (exercise04)"
```

Selecione **Python (exercise04)** no Jupyter ou no VS Code. Para conferir o ambiente:

```bash
python -c "import torch, scipy, plotly, anywidget; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.is_available())"
```

## 3. Executar os dois problemas em uma única execução

```bash
jupyter lab exercise04.ipynb
```

Selecione **Python (exercise04 — AMD ROCm)** e utilize **Run All**. Com `annotations_big.csv` e `annotations_small.csv` já presentes, nenhuma interação é necessária: a seção 3 valida os arquivos e mostra os pontos sobre a imagem. Os CSVs originais são somente lidos.

Cada arquivo de origem contém 200 pontos de fratura e 200 de fundo. A composição preserva todos os registros:

| Problema | Fundo (0) | Fratura/grande (1) | Pequena (2) | Total |
|---|---:|---:|---:|---:|
| Binário | 400 | 400 | — | 800 |
| Multiclasse | 400 | 200 | 200 | 800 |

No multiclasse, o rótulo 1 identifica as fraturas de `big` e o rótulo 2, as de `small`. Os dois fundos são concatenados. No binário, ambas as seleções de fratura recebem rótulo 1. Não é necessário criar `annotations_binary.csv` ou `annotations_multiclass.csv` na pasta do notebook; os dois conjuntos são derivados automaticamente e gravados em `results/main/datasets/`, com as colunas de rastreabilidade `source` e `source_row`.

Os experimentos estão organizados em blocos autocontidos:

| Bloco | Experimentos |
|---|---|
| Seção 5 — Binário | Base com 10 épocas; seleção `big`; seleção `small`; arquitetura ampliada; 100 épocas; validação; regularização e parada antecipada |
| Seção 6 — Multiclasse | Base com 10 épocas; arquitetura ampliada; 100 épocas; validação; regularização e parada antecipada |

Cada bloco inicializa seus próprios dados, configurações, modelos e resumos. A multiclasse não reutiliza modelos nem variáveis de treinamento do bloco binário. Após as seções comuns 1 a 4, é possível executar somente o bloco desejado; **Run All** executa ambos.

Para execução pelo terminal:

```bash
jupyter nbconvert \
  --to notebook \
  --execute exercise04.ipynb \
  --output exercise04.executed.ipynb \
  --ExecutePreprocessor.timeout=3600
```

Esse comando preserva o notebook de entrada e grava suas saídas em `exercise04.executed.ipynb`. Para atualizar o próprio notebook, substitua `--output exercise04.executed.ipynb` por `--inplace`. Os resultados em `results/main/` são atualizados nas novas execuções; os arquivos originais de picking não são sobrescritos.

## 4. Reutilização e proteção do picking

A seção 3 verifica separadamente `annotations_big.csv` e `annotations_small.csv`:

- se o arquivo existe, valida as 200 fraturas e os 200 fundos e exibe a seleção sobre a imagem;
- se o arquivo está ausente, abre o anotador apenas para esse conjunto;
- se um CSV existir no momento de salvar, o anotador recusa a sobrescrita, inclusive a partir de um widget aberto anteriormente.

Na primeira seleção de um arquivo ausente, os pontos de fratura são escolhidos antes dos pontos de fundo. Depois de clicar em **Salvar CSV**, continue na próxima célula. O formato é `x,z,label`, com `x=coluna`, `z=linha`, índices começando em zero e classes `0=fundo`, `1=fratura`. Os CSVs já existentes não precisam ser refeitos.

A concatenação não remove registros repetidos entre os dois arquivos; todas as entradas são preservadas. Rótulos contraditórios para o mesmo centro são recusados pela validação. O conteúdo dos CSVs originais é identificado por SHA-256 e conferido novamente ao final da execução.

## 5. Configurações e saída por problema

As configurações comuns estão na primeira célula: sementes, épocas, lotes, dispositivo e stride. `INFERENCE_STRIDE = 1` classifica todos os pixels; `16` ou `32` produzem prévias em grade reduzida, sem reduzir o treino. O diretório padrão é `results/main/`, com sobrescrita dos resultados autorizada.

As seções 5 e 6 definem `binary_config` e `multiclass_config`, respectivamente. A base mantém canais `(6, 12)`, passo 1, lotes de 20 e 10 épocas. A arquitetura ampliada usa `(12, 24)`. O treinamento prolongado usa 100 épocas. O experimento controlado acrescenta `weight_decay=0.001`, paciência de 10 épocas e restauração dos pesos da melhor época de validação.

A multiclasse mantém as três saídas sigmoides, alvos one-hot, erro quadrático e decisão por argmax, sem reamostragem nem pesos adicionais por classe. Há 400 fundos e 200 exemplos de cada tipo de fratura.

## Fidelidade à conversão

| MATLAB | Python | Convenção |
|---|---|---|
| `padimage.m` | `PatchExtractor` | Borda replicada |
| `interp2` | Interpolação bilinear com `align_corners=True` | `17 × 17` → `32 × 32` |
| `cnnsetup.m` | `CrackCNN` | Xavier uniforme, bias zero |
| `cnnff.m` | `CrackCNN.forward` | Convolução válida, sigmoide, pooling por média e saída sigmoide |
| `cnnbp.m` | `matlab_loss` e autograd | `0.5 * soma((saída - alvo)²) / N` |
| `cnnapplygrads.m` | SGD | Passo 1, sem momentum |
| `cnntrain.m` | `train_cnn` | Lotes de 20, embaralhamento por época e registro de perda |

A CNN de referência tem **2.570 parâmetros**: convoluções `1→6→12` com kernels `5 × 5`, pooling médio `2 × 2` após cada convolução e camada final de `300→2`. A imagem `457 × 977` mantém sua escala original `[-1, 1]`. A previsão usa o maior escore das saídas, que não são probabilidades normalizadas.

A implementação preserva a inversão dos kernels de `convn` e a vetorização por colunas dos mapas MATLAB. A ordem das saídas passa de `[fratura, fundo]` para `[fundo, fratura]`, coerente com os IDs `0,1`. Na inferência, a indexação NumPy por linhas é usada de forma consistente para reconstruir o mapa; não se aplicam os `reshape` por colunas nem os limites de blocos fixos do script original.

A imagem é carregada do pacote oficial, mas os experimentos deste roteiro utilizam os conjuntos manuais combinados. Os pontos oficiais permanecem disponíveis apenas para consulta. Nos experimentos de validação, todas as ocorrências de um centro ficam no mesmo subconjunto. Patches vizinhos ainda podem se sobrepor, e ambos os subconjuntos pertencem à mesma imagem; trata-se de validação interna, não de teste independente.

As sementes são fixas, mas MATLAB e PyTorch usam geradores aleatórios diferentes. A conversão usa float32 em lugar de double. Portanto, a reprodução mantém o procedimento e não exige igualdade numérica com uma execução MATLAB. O registro `rL` contém um valor por atualização, equivalente a `net.rL(2:end)`; o MATLAB duplica a primeira perda ao inicializar esse vetor. Quando a divisão de validação gera um último lote incompleto, o Python o processa; o código MATLAB original exige divisibilidade por 20.

## Coleta dos resultados

Cada execução atualiza `results/main/`, preservando os CSVs originais de anotação. A estrutura das novas saídas é:

```text
results/main/
├── settings.json
├── analysis/
├── datasets/
│   ├── annotations_binary.csv
│   └── annotations_multiclass.csv
├── binary/
│   ├── dataset.csv
│   ├── summary.csv
│   ├── map_summary.csv
│   └── <experimento>/
├── multiclass/
│   ├── dataset.csv
│   ├── summary.csv
│   ├── map_summary.csv
│   └── <experimento>/
├── summary.csv
└── map_summary.csv
```

A seção 7.1 lê os resultados existentes e exporta em `analysis/` os diagnósticos de margens dos escores, distribuição regional, sobreposição dos patches, seleção de épocas, erros multiclasse, métricas com classes reunidas, concordância dos mapas e validação nos centros comuns. Também salva uma figura comparativa dos modelos de 100 épocas. Essa célula não reexecuta o treinamento.

Os resumos da raiz distinguem os dois problemas pela coluna `task`. Os diretórios de experimentos de execuções anteriores, fora desses dois blocos, não são utilizados pelo roteiro atual. Cada experimento tem sua própria subpasta com:

- `history.csv` e `batches.csv`: perdas, acurácias por época e média móvel por atualização;
- `metrics.csv`, `class_metrics.csv` e `confusion_*.csv`: métricas do treino e, quando houver, da validação;
- `annotations.csv`: pontos efetivamente usados e identificação do subconjunto;
- `config.json` e `model.pt`: configuração, versões, dispositivo, tempos, época selecionada e pesos;
- `maps.npz`: rótulos, escores, eixos `x,z`, stride e dimensões da imagem original;
- `labels.png`: IDs inteiros das classes; use `classification.png` para visualizar em cores;
- `map_distribution.csv`: contagem e fração dos rótulos previstos;
- figuras PNG de curvas, matrizes de confusão, mapas e escores.

Para ler um mapa:

```python
from pathlib import Path
import numpy as np

run_dir = Path("results/main")
maps = np.load(run_dir / "binary/reference_10epochs/maps.npz")
labels = maps["labels"]
crack_score = maps["scores"][:, :, 1]
x, z = maps["x"], maps["z"]
```

Com stride 1, `labels` tem forma `(457, 977)` e contém um rótulo por pixel. Com stride maior, os arrays representam somente os centros amostrados. A referência completa é obtida usando stride 1. O tempo depende do dispositivo e do tamanho dos lotes; `INFERENCE_BATCH_SIZE` pode ser reduzido para diminuir o uso de memória.

## Solução de problemas

- **Download indisponível:** baixe [CNN_lab.zip](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.CNN/Chapter.CNNCrack/CNN_lab.zip) pelo navegador e salve em `exercise04/.cache/CNN_lab.zip`. O SHA-256 será verificado.
- **Anotador não aparece:** atualize o ambiente pelo YAML, reinicie o kernel e confirme **Python (exercise04)**. A seleção precisa do JupyterLab ou VS Code com suporte a widgets.
- **CSV ainda não existe:** conclua a seleção do arquivo ausente e clique em **Salvar CSV** antes de continuar. Com os dois CSVs de origem disponíveis, a execução não exige interação.
- **Memória da GPU insuficiente:** reduza `INFERENCE_BATCH_SIZE` ou selecione `DEVICE = "cpu"` e reinicie a execução. Os patches da imagem inteira não são materializados juntos.
- **Kernel não encontrado:** repita o registro da seção 2 com o ambiente `exercise04` ativo.
- **Função nova de `crack_cnn` não é reconhecida:** reexecute a primeira célula atualizada; ela recarrega o módulo local antes de importar as rotinas. Reiniciar o kernel também limpa a versão anterior em memória.

## Fontes e atribuição

- [Enunciado e pacote MATLAB CNN_lab — Shihang Feng](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.CNN/Chapter.CNNCrack/lab.html).
- Gerard T. Schuster, *Machine Learning Methods in Geoscience*, seção 10.5 e apêndice 10.18; PDF disponível na raiz deste repositório.
- Rotinas CNN de Rasmus Berg Palm, copyright 2012; licença original preservada em `LICENSE-CNN_lab`.
- [Interpolação no PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html).

## Autor

[![Avatar de Lucas Torres Marques no GitHub](https://github.com/Lucastmarques.png?size=160)](https://github.com/Lucastmarques)

[**Lucas Torres Marques**](https://github.com/Lucastmarques)
