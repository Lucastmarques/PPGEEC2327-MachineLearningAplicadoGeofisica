# Exercício 02 — Classificação com redes neurais

Este diretório contém a conversão para Python/NumPy dos laboratórios MATLAB da seção 4.8 do livro *Machine Learning Methods in Geoscience*.

O trabalho cobre:

- classificação binária com resultados semelhantes às Figuras 4.13, 4.15, 4.17 e 4.18;
- validação do gradiente analítico por diferenças finitas, como na Figura 4.26;
- classificação multiclasse pela quantidade de bits iguais a 1, como no exemplo da seção 4.4.2 indicado no enunciado;
- discussão dos resultados, das escolhas numéricas e das diferenças encontradas nos fontes MATLAB.

O notebook já está executado e contém as tabelas e figuras. Todas as saídas podem ser regeneradas de forma determinística.

## Arquivos

- `exercise02.ipynb`: relatório completo, experimentos, gráficos e discussão;
- `neural_network.py`: conversão reutilizável das rotinas MATLAB de geração de dados, propagação direta, retropropagação, treinamento e validação do gradiente;
- `conda.yaml`: ambiente Conda necessário para executar o notebook;
- `README.md`: este guia.

Os anexos originais presentes em `NN_ErSE294/` foram usados como referência, mas permanecem ignorados pelo Git conforme a configuração existente do repositório.

## Correspondência com os códigos MATLAB

| MATLAB | Python | Finalidade |
|---|---|---|
| `NNode1.m` / `Node1.m` | `train_backtracking` e funções auxiliares | Classificação binária com busca de passo por bisseção |
| `NNode_test.m` | `gradient_check` e `finite_difference_gradients` | Comparação dos gradientes analítico e numérico |
| `NNode_multiclass.m` | `train_conjugate_gradient` | Classificação multiclasse com softmax e gradientes conjugados |
| `gradientnn.m` | `loss_and_gradients` | Propagação direta, função de perda e retropropagação |
| `activation.m` | `activate` | Sigmoide, ReLU, ELU e softmax |

A implementação usa apenas NumPy no núcleo. O SciPy fornece o método de gradientes conjugados empregado no experimento multiclasse, e Matplotlib/Pandas produzem as figuras e tabelas.

## 1. Criar o ambiente

Com Conda, Miniconda ou Mambaforge instalado, entre neste diretório:

```bash
cd exercise02
```

Crie e ative o ambiente:

```bash
conda env create --file conda.yaml
conda activate exercise02
```

Para sincronizar um ambiente já existente:

```bash
conda env update --file conda.yaml --prune
```

## 2. Registrar o kernel Jupyter

```bash
python -m ipykernel install --user \
  --name exercise02 \
  --display-name "Python (exercise02)"
```

No Jupyter ou no VS Code, selecione **Python (exercise02)**.

## 3. Executar o notebook

### JupyterLab

```bash
jupyter lab exercise02.ipynb
```

### Jupyter Notebook

```bash
jupyter notebook exercise02.ipynb
```

Depois, use **Run > Run All Cells**. Pelo terminal, é possível atualizar o próprio arquivo com todas as saídas:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace exercise02.ipynb \
  --ExecutePreprocessor.timeout=600
```

## Resultados reproduzíveis

Com as sementes registradas no notebook, os resultados esperados são:

| Experimento binário | Arquitetura | Perda | Ativação oculta | Perda final | Acurácia de treinamento |
|---|---|---|---|---:|---:|
| Figura 4.13 | `[5, 1]` | L2/MSE | — | `0.0155444` | `100%` |
| Figura 4.15 | `[5, 1, 1]` | L2/MSE | Sigmoide | `0.0177386` | `100%` |
| Figura 4.17 | `[5, 1, 1]` | Entropia cruzada | Sigmoide | `0.0238862` | `100%` |
| Figura 4.18 | `[5, 1, 1]` | Entropia cruzada | ReLU | `0.00864937` | `100%` |

Na validação do gradiente da rede `[5, 10, 1]`:

- maior erro absoluto: aproximadamente `2.53e-11`;
- erro relativo L2: aproximadamente `9.97e-11`.

Na classificação multiclasse `[5, 1, 6]`, tanto sigmoide quanto ELU atingem `100%` de acurácia de treinamento. Após 200 iterações registradas, as perdas finais são aproximadamente `0.08231` e `0.000487`, respectivamente.

Os valores podem sofrer diferenças mínimas entre versões de NumPy/SciPy, mas as conclusões e acurácias devem permanecer as mesmas.

## Observações sobre os fontes originais

- O eixo vertical dos gráficos MATLAB é chamado de “RMS”, porém a versão oficial de `gradientnn.m` usada no laboratório calcula a média do erro quadrático sem aplicar a raiz. O notebook usa o nome L2/MSE para evitar essa ambiguidade.
- O `activation.m` do pacote multiclasse chama a opção 2 de ReLU em mensagens e legendas, mas a fórmula implementada é ELU com `alpha=1`. A conversão preserva essa fórmula sob o nome correto `elu` e oferece também uma ReLU verdadeira para o experimento binário.
- A acurácia reportada é calculada sobre os próprios dados de treinamento, como no laboratório. Ela não deve ser interpretada como estimativa de generalização.
- O código usa sementes fixas porque os geradores aleatórios de MATLAB e NumPy não produzem as mesmas sequências.

## Fontes

- [Descrição oficial do laboratório multiclasse](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.FCN/Chapter.MultiGNN/lab.html)
- [Descrição oficial do laboratório de validação do gradiente](http://utam.gg.utah.edu/Classes/ML/LAB1/Chapter.Book.FCN/Chapter.Gradient.Neural/lab.html)
- `Machine Learning Methos in Geoscience.pdf`, disponível na raiz deste repositório

## Solução de problemas

### O módulo `neural_network` não foi encontrado

Abra o notebook a partir do diretório `exercise02` ou mantenha a estrutura do repositório. A primeira célula de código também reconhece a execução a partir da raiz.

### O kernel não aparece

```bash
conda activate exercise02
python -m ipykernel install --user --name exercise02 --display-name "Python (exercise02)"
```

### O SciPy encerra com “maximum number of iterations exceeded”

O limite de 200 iterações é deliberado para reproduzir o orçamento do laboratório. O resultado continua válido quando a perda, a acurácia e os gráficos coincidem com os valores esperados; a mensagem significa apenas que o critério interno de tolerância do otimizador é mais rigoroso que esse orçamento.

## Autor

[![Avatar de Lucas Torres Marques no GitHub](https://github.com/Lucastmarques.png?size=160)](https://github.com/Lucastmarques)

[**Lucas Torres Marques**](https://github.com/Lucastmarques)
