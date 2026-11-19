# PPGEEC2327 — Tópicos Especiais em Processamento Inteligente da Informação

> **Turma:** T01  
> **Período:** 2026.2  
> **Horário:** 5T2345

## Objetivo do repositório

Este repositório tem como objetivo armazenar as soluções dos exercícios e das atividades desenvolvidas na disciplina **PPGEEC2327 — Tópicos Especiais em Processamento Inteligente da Informação — T01 (2026.2 — 5T2345)**.

A disciplina está sendo cursada durante o Doutorado no **Programa de Pós-Graduação em Engenharia Elétrica e de Computação (PPgEEC)**, vinculado ao **Departamento de Engenharia de Computação e Automação (DCA)** da **Universidade Federal do Rio Grande do Norte (UFRN)**.

## Docente

A disciplina é ministrada pelo Prof. Dr. [**Tiago Tavares Leite Barros**](http://www.docente.ufrn.br/tiago.barros "Acesse a página pública deste docente").

## Organização do repositório

Cada exercício ou atividade possui um diretório próprio e autocontido. Cada pasta reúne os materiais necessários para compreender a solução e, quando aplicável, configurar o ambiente e executar os experimentos, incluindo:

- um `README.md` com a descrição da atividade e dos arquivos;
- artigo e slides em PDF, no caso de apresentações;
- instruções para instalação das dependências, quando necessárias;
- orientações para execução dos scripts e/ou notebooks Jupyter, quando houver;
- arquivos de configuração do ambiente, quando necessários;
- códigos-fonte, notebooks, dados e demais recursos específicos da atividade.

A estrutura geral segue o formato:

```text
.
├── README.md
├── exercise01/
│   ├── README.md
│   ├── conda.yaml
│   └── exercise01.ipynb
├── exercise02/
│   ├── README.md
│   ├── conda.yaml
│   ├── exercise02.ipynb
│   └── neural_network.py
├── exercise03/
│   ├── README.md
│   ├── annotations.csv
│   ├── conda.yaml
│   └── exercise03.ipynb
├── exercise04/
│   ├── README.md
│   ├── conda.yaml
│   ├── exercise04.ipynb
│   ├── crack_cnn.py
│   ├── annotation_picker.py
│   ├── annotations_big.csv
│   ├── annotations_small.csv
│   └── LICENSE-CNN_lab
├── exercise06/
│   ├── README.md
│   ├── exercise06.ipynb
│   ├── conda.yaml
│   ├── src/
│   └── original/
└── Presentation01/
    ├── README.md
    ├── swell_noise_ufrj.pdf
    └── Apresentação de artigo em ML aplicado à geofísica (Lucas e Rosélia).pdf
```

O README de cada atividade descreve seus materiais e orientações de consulta. Para os exercícios com código, essas instruções têm precedência para instalação e execução, pois podem utilizar dependências ou ambientes distintos.

## Atividades disponíveis

| Diretório | Conteúdo |
|---|---|
| [`exercise01`](./exercise01/) | Comparação dos otimizadores Momentum, Nesterov Accelerated Gradient (NAG) e Adam na minimização da função de Rosenbrock. |
| [`exercise02`](./exercise02/) | Classificação binária e multiclasse com redes neurais implementadas em NumPy, incluindo validação numérica da retropropagação e reprodução dos experimentos do Capítulo 4 de *Machine Learning Methods in Geoscience*. |
| [`exercise03`](./exercise03/) | Comparação de FCNN e SVM-RBF na identificação de refletores e filtragem de artefatos de migração em uma imagem sísmica, reproduzindo o laboratório dos capítulos 5 e 7 de *Machine Learning Methods in Geoscience*. |
| [`exercise04`](./exercise04/) | Conversão do laboratório MATLAB CNN_lab para Python/PyTorch, com classificação binária e multiclasse em uma única execução, reutilização do picking manual, variação da arquitetura e comparação de 10 e 100 épocas. |
| [`exercise06`](./exercise06/) | Conversão do laboratório CSC para NumPy/SciPy, com notebook executado, ADMM para treinamento e reconstrução, variações de tamanho/K/L1 e comparação visual com filtros CNN. |
| [`Presentation01`](./Presentation01/) | Apresentação em dupla de artigo sobre aprendizado profundo para atenuação de ruído de swell em dados sísmicos marinhos, com o artigo e os slides em PDF. |

Esta tabela será atualizada conforme novas atividades forem adicionadas ao repositório.

## Autor

[![Avatar de Lucas Torres Marques no GitHub](https://github.com/Lucastmarques.png?size=160)](https://github.com/Lucastmarques)

[**Lucas Torres Marques**](https://github.com/Lucastmarques)

## Colaboradores

As atividades em dupla contam com a participação de colaboradores, identificados abaixo com suas respectivas contribuições.

[![Avatar de Rosélia Nascimento no GitHub](https://github.com/roseliasilva.png?size=160)](https://github.com/roseliasilva)

[**Rosélia Nascimento**](https://github.com/roseliasilva)

Participou da preparação da [Apresentação 01 — Artigo em ML aplicado à geofísica](./Presentation01/README.md), sobre o uso de aprendizado profundo para atenuar ruído de *swell* em dados sísmicos marinhos.
