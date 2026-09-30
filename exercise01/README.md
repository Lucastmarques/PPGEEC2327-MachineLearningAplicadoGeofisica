# Exercício 01 — Comparação de otimizadores

Este diretório contém um experimento de minimização da função não linear de Rosenbrock utilizando implementações próprias, em PyTorch, dos seguintes otimizadores:

- Momentum Gradient Descent;
- Nesterov Accelerated Gradient (NAG);
- Adam.

O notebook registra e compara as trajetórias dos três métodos sobre as curvas de nível da função. As respostas e figuras já estão salvas no arquivo, mas também é possível recriar todo o ambiente e executar todas as células seguindo as instruções abaixo.

## Arquivos

- `exercise01.ipynb`: notebook com as implementações, experimentos, gráficos e conclusão;
- `conda.yaml`: especificação do ambiente Conda e das dependências;
- `README.md`: este guia de instalação e execução.

## Requisitos

- Linux ou WSL com uma distribuição Linux compatível;
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), Miniconda ou Mambaforge;
- para aceleração por GPU AMD: GPU, driver e runtime compatíveis com ROCm 7.2.

O `conda.yaml` instala a distribuição oficial do PyTorch 2.13.0 para ROCm 7.2. Essa distribuição é voltada a Linux. O notebook também pode ser executado pela CPU caso uma GPU AMD compatível não esteja disponível, embora a instalação continue utilizando o pacote ROCm.

## 1. Instalar o Conda

Se `conda` ainda não estiver instalado, utilize o instalador do Miniconda descrito na [documentação oficial](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). Depois da instalação, feche e abra novamente o terminal e confirme:

```bash
conda --version
```

Se o comando não for reconhecido, inicialize o Conda para o shell Bash e abra um novo terminal:

```bash
conda init bash
```

## 2. Criar o ambiente a partir do `conda.yaml`

No terminal, entre neste diretório:

```bash
cd exercise01
```

Crie o ambiente. Esse processo instala Python, Jupyter, PyTorch, NumPy, Pandas, Matplotlib, Seaborn e Plotly:

```bash
conda env create --file conda.yaml
```

Ative o ambiente criado:

```bash
conda activate exercise01
```

Para atualizar um ambiente já existente após alterações no YAML, use:

```bash
conda env update --file conda.yaml --prune
```

## 3. Registrar o kernel do Jupyter

Com o ambiente `exercise01` ativo, registre-o como um kernel disponível:

```bash
python -m ipykernel install --user --name exercise01 --display-name "Python (exercise01)"
```

No Jupyter ou no VS Code, selecione o kernel **Python (exercise01)** antes de executar as células.

## 4. Verificar o PyTorch e a GPU AMD

Execute:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('ROCm:', torch.version.hip); print('GPU disponível:', torch.cuda.is_available())"
```

Mesmo em GPUs AMD, o PyTorch utiliza a interface `torch.cuda` para consultar e acessar a GPU. Um resultado esperado com aceleração habilitada é semelhante a:

```text
PyTorch: 2.13.0+rocm7.2
ROCm: 7.2
GPU disponível: True
```

Se `GPU disponível` for `False`, o notebook ainda poderá executar na CPU. Nesse caso, verifique a compatibilidade da GPU, do driver e da instalação ROCm.

## 5. Abrir o notebook

Com o ambiente ativo, escolha uma das interfaces abaixo.

### JupyterLab

```bash
jupyter lab exercise01.ipynb
```

### Jupyter Notebook clássico

```bash
jupyter notebook exercise01.ipynb
```

### VS Code

Abra `exercise01.ipynb`, clique no seletor de kernel no canto superior direito e escolha **Python (exercise01)**.

## 6. Executar o notebook inteiro

Na interface do Jupyter, utilize **Run > Run All Cells**. No VS Code, utilize **Run All**.

Também é possível executar todas as células pelo terminal e salvar o resultado em um novo arquivo:

```bash
jupyter nbconvert \
  --to notebook \
  --execute exercise01.ipynb \
  --output exercise01.executed.ipynb \
  --ExecutePreprocessor.timeout=600
```

Essa opção preserva o notebook original. Para atualizar o próprio `exercise01.ipynb` com as novas saídas, use `--inplace` conscientemente:

```bash
jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace exercise01.ipynb \
  --ExecutePreprocessor.timeout=600
```

## Resultados esperados

Com os parâmetros definidos no notebook e nove atualizações, os erros finais são aproximadamente:

| Otimizador | Erro final |
|---|---:|
| Momentum | `0.588399` |
| NAG | `0.131297` |
| Adam | `0.145565` |

Pequenas diferenças podem ocorrer por versão da biblioteca, tipo numérico ou plataforma. O notebook também gera gráficos individuais e uma comparação lado a lado das três trajetórias.

## Solução de problemas

### O kernel `exercise01` não aparece

Ative o ambiente e registre novamente o kernel:

```bash
conda activate exercise01
python -m ipykernel install --user --name exercise01 --display-name "Python (exercise01)"
```

### O módulo `torch` ou `matplotlib` não foi encontrado

Confirme que o ambiente correto está ativo:

```bash
conda activate exercise01
which python
python -c "import torch, matplotlib; print('Dependências carregadas')"
```

### As versões instaladas não correspondem ao `conda.yaml`

Um ambiente antigo com o mesmo nome pode conter pacotes diferentes. Sincronize-o com o arquivo atual:

```bash
conda activate exercise01
conda env update --file conda.yaml --prune
```

Em seguida, execute novamente o comando de verificação do PyTorch. Se as versões ainda não coincidirem, remova e recrie o ambiente conforme a seção abaixo.

### A criação do ambiente foi interrompida

Execute novamente a atualização do ambiente:

```bash
conda env update --file conda.yaml --prune
```

### Remover o ambiente

Caso seja necessário recriá-lo do zero:

```bash
conda deactivate
conda env remove --name exercise01
conda env create --file conda.yaml
```

## Autor

[![Avatar de Lucas Torres Marques no GitHub](https://github.com/Lucastmarques.png?size=160)](https://github.com/Lucastmarques)

[**Lucas Torres Marques**](https://github.com/Lucastmarques)
