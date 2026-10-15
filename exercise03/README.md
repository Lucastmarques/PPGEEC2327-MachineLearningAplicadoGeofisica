# Exercício 03 — Filtragem de ruídos de migração com FCNN e SVM-RBF

Este diretório contém uma reprodução em Python do laboratório **Identifying Migration Artifacts and Reflectors by Machine Learning Methods**, associado aos capítulos 5 e 7 do livro *Machine Learning Methods in Geoscience*.

O objetivo é comparar uma rede neural totalmente conectada (FCNN) e uma máquina de vetores de suporte com kernel de função de base radial (SVM-RBF) na classificação dos pixels de uma imagem sísmica migrada como:

- `1`: refletor/sinal a preservar;
- `0`: artefato ou fundo a remover.

O notebook já contém uma execução de referência com figuras, métricas e discussões. Na primeira execução local, as coordenadas são selecionadas interativamente e salvas em `annotations.csv`; depois disso, o arquivo é reutilizado automaticamente.

## Arquivos

- `exercise03.ipynb`: implementação completa, resultados e análise;
- `conda.yaml`: ambiente Conda necessário para reproduzir a execução;
- `README.md`: instruções e descrição do experimento.
- `annotations.csv`: criado na primeira execução com os 80 pontos escolhidos manualmente.

Não são necessários arquivos MATLAB nem dados sísmicos versionados no repositório. O notebook baixa o pacote oficial `Comparision.zip` diretamente do laboratório e lê `mig_and_features.mat` em memória. Depois de selecionar os pontos, mantenha `annotations.csv` junto ao notebook e, se desejar reproduzir exatamente sua interpretação em outra máquina, versione também esse arquivo.

O ZIP oficial é persistido em `exercise03/.cache/Comparision.zip`. A cada execução, o SHA-256 do cache é verificado antes do uso. Se o arquivo estiver ausente ou inválido, o notebook realiza um novo download e substitui o cache de forma atômica. A pasta `.cache` é local e está excluída do Git.

## Dados e correspondência com o MATLAB

O arquivo oficial contém quatro matrizes `201 × 201`:

| Variável MATLAB | Conteúdo |
|---|---|
| `mig` | Imagem sísmica migrada com refletores e artefatos |
| `cohe` | Coerência local |
| `angl` | Ângulo de mergulho local |
| `engy` | Amplitude/envelope local |

As três características são normalizadas exatamente como no script `comapre_main.m`:

```text
(atributo - média) / (máximo - mínimo)
```

Assim como o MATLAB, o notebook permite selecionar os pontos diretamente sobre a imagem. Para evitar o bloqueio e o canvas branco produzido por `ginput` em alguns ambientes, a interface usa um anotador Plotly não bloqueante. Quando `annotations.csv` ainda não existe, ele solicita, nesta ordem:

1. 50 pontos de artefato/fundo, rotulados como `0`;
2. 30 pontos de refletor, rotulados como `1`.

As coordenadas são arredondadas para o pixel mais próximo, verificadas quanto a quantidade, unicidade e limites e gravadas no CSV. Em todas as execuções seguintes, o notebook carrega esse arquivo em vez de abrir novamente a ferramenta de seleção.

## Modelos reproduzidos

### FCNN

- três atributos originais expandidos para nove termos lineares e quadráticos;
- uma camada oculta de nove neurônios sigmoides;
- uma saída sigmoide;
- entropia cruzada binária;
- gradiente em lote completo;
- 500 atualizações;
- passo inicial igual a `10`, reduzido por um fator `0,8` quando a perda aumenta;
- sem regularização, como na configuração do laboratório.

### SVM-RBF

- três atributos normalizados;
- `C = 1`;
- largura gaussiana `sigma = 0,1`;
- `gamma = 1/(2 sigma²) = 50`, forma equivalente à função `gaussianKernel.m`.

Não são realizados ajuste de hiperparâmetros ou validação cruzada, pois o objetivo é reproduzir a configuração do laboratório.

## 1. Criar o ambiente

Com Conda, Miniconda ou Mambaforge instalado:

```bash
cd exercise03
conda env create --file conda.yaml
conda activate exercise03
```

Para atualizar um ambiente existente:

```bash
conda env update --file conda.yaml --prune
```

## 2. Registrar o kernel

```bash
python -m ipykernel install --user \
  --name exercise03 \
  --display-name "Python (exercise03)"
```

No Jupyter ou no VS Code, selecione **Python (exercise03)**.

Se o anotador não aparecer, o ambiente provavelmente foi criado antes da inclusão de `plotly`, `anywidget` e `ipywidgets`. Atualize-o e reinicie completamente o kernel:

```bash
conda env update --file conda.yaml --prune
```

Depois confirme que o kernel selecionado continua sendo **Python (exercise03)**. Apenas instalar os pacotes sem reiniciar o kernel não ativa os novos widgets na sessão que já estava aberta.

## 3. Executar o notebook

### JupyterLab

```bash
jupyter lab exercise03.ipynb
```

### Jupyter Notebook

```bash
jupyter notebook exercise03.ipynb
```

Execute as células até a seção de anotação. Na primeira execução:

1. clique na imagem até completar 50 artefatos;
2. o anotador muda automaticamente para a classe refletor; selecione mais 30 pontos;
3. use os botões **Desfazer** ou **Limpar tudo**, se necessário;
4. revise os marcadores e clique em **Salvar CSV**;
5. depois da confirmação, execute a próxima célula.

Cada clique adiciona um ponto no pixel mais próximo. A contagem e a classe ativa aparecem acima da imagem. Para escolher tudo novamente depois de salvar, remova ou renomeie `annotations.csv` e execute novamente a seção de anotações.

Depois que o CSV existir, também é possível executar de forma não interativa pelo terminal:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace exercise03.ipynb \
  --ExecutePreprocessor.timeout=600
```

A primeira execução deve ser feita no JupyterLab ou no VS Code, pois requer interação com as figuras. Quando o cache ainda não existe, ela também requer conexão com a internet para baixar aproximadamente 2 MB do site do laboratório. O comando `nbconvert` pressupõe que `annotations.csv` já foi criado.

## Resultados esperados

Na execução de referência salva no notebook:

- a perda da FCNN cai de aproximadamente `0,6635` para `0,0093`;
- ambos acertam os 80 pontos anotados: acurácia, precisão, recall e F1 iguais a `1` no treinamento;
- a FCNN preserva `6,29%` dos pixels e `99,43%` da energia; o SVM preserva `11,38%` dos pixels e `99,48%` da energia;
- o SVM utiliza 33 vetores de suporte;
- a concordância entre máscaras é de `94,91%`, mas a IoU das regiões preservadas é de `55,25%`.

Na inspeção visual, a FCNN parece preservar os refletores principais com menos regiões espúrias. O SVM mantém algumas estruturas no topo que interpreto como possíveis falsos positivos. No trecho entre aproximadamente `(x,z)=(100,125)` e `(175,175)`, ambos podem remover segmentos de refletores. Essas observações motivam a discussão no notebook, mas ainda precisam de rótulos independentes para confirmar os tipos de erro.

Seus resultados podem mudar de acordo com os pontos escolhidos. Esses números são métricas sobre os pontos usados no treinamento e descrições da máscara completa. Não constituem uma estimativa de generalização, pois o laboratório não fornece rótulos de referência para todos os pixels nem um conjunto de teste separado.

## Reprodutibilidade e integridade

O notebook valida o pacote baixado com o SHA-256:

```text
e8618f468c32baa06f714e73c2a02bac5b5e399d44fd70666119219a4c606e6a
```

Se o mantenedor do laboratório substituir o ZIP por uma nova versão, a verificação falhará deliberadamente. Nesse caso, a nova versão deve ser inspecionada antes de atualizar o hash.

Para forçar um novo download sem alterar as anotações, remova apenas `exercise03/.cache/Comparision.zip` e execute novamente a seção de aquisição dos dados.

## Limitações

- A escolha dos pontos é interpretativa e influencia as fronteiras aprendidas.
- As métricas de treinamento tendem a ser otimistas.
- A energia preservada é uma medida descritiva: artefatos e refletores podem contribuir simultaneamente para um mesmo pixel.
- Regiões onde sinal e artefato se cruzam são difíceis de separar usando apenas três atributos locais.
- Os hiperparâmetros são os do laboratório e não necessariamente os melhores para esta anotação.

## Fonte

- [Laboratório oficial de SVM e redes neurais](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.SVM/lab.html)
- [Pacote oficial `Comparision.zip`](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.SVM/Comparision.zip)

## Autor

[![Avatar de Lucas Torres Marques no GitHub](https://github.com/Lucastmarques.png?size=160)](https://github.com/Lucastmarques)

[**Lucas Torres Marques**](https://github.com/Lucastmarques)
