# LLM-Wiki vs RAG

Projeto de Iniciação Científica voltado à avaliação e comparação entre **LLM-Wiki** e **RAG (Retrieval-Augmented Generation)** para construção e consulta de bases de conhecimento verificáveis.

## Sobre o projeto

O projeto busca investigar em quais cenários uma base de conhecimento construída e mantida por LLMs, organizada em páginas Markdown, é suficiente para responder perguntas e em quais situações a recuperação dinâmica de informações externas por meio de RAG se torna necessária.

Entre os principais aspectos que serão avaliados estão:

- Acurácia das respostas
- Atualidade das informações
- Consistência
- Fidelidade às informações disponíveis
- Capacidade de citar fontes
- Robustez para diferentes tipos de consultas
- Comportamento com o aumento da quantidade de informações
- Limitações e pontos de falha de cada abordagem

## Status do projeto

Atualmente, o projeto está **entre a terceira e a quarta semana de desenvolvimento**.

Neste momento, ainda estamos na **fase de estudos e testes iniciais**, antes do início efetivo da implementação e avaliação do projeto.

### Estudos e testes iniciais

Até o momento, foram realizados estudos e experimentos preliminares envolvendo:

- Testes com o modelo **Gemma 3 270M**
- Testes de perguntas e respostas utilizando informações presentes em notícias
- Análise inicial do comportamento do LLM ao responder perguntas utilizando apenas um conjunto de informações fornecidas no contexto
- Estudo inicial da abordagem **RAG**
- Divisão de documentos em pequenos trechos (chunks)
- Geração de embeddings utilizando o modelo **multilingual-e5-base**
- Recuperação dos chunks mais relevantes para uma determinada pergunta
- Busca por similaridade entre a pergunta e os embeddings dos documentos
- Testes iniciais utilizando notícias como base documental

Os primeiros experimentos de RAG foram desenvolvidos utilizando `SentenceTransformer`, `RecursiveCharacterTextSplitter` e cálculo de similaridade entre os vetores gerados para as perguntas e os documentos. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

Também foram realizados testes utilizando o Gemma 3 270M para responder perguntas com base nas informações fornecidas diretamente ao modelo. :contentReference[oaicite:4]{index=4}

Esses experimentos fazem parte da etapa inicial de familiarização com as tecnologias e de preparação para a implementação dos experimentos principais.

## Próximas etapas

Após a conclusão dessa fase inicial de estudos e testes, o projeto avançará para a construção das abordagens de **LLM-Wiki e RAG**, definição da base documental e elaboração do conjunto de perguntas que será utilizado na avaliação.

Posteriormente, serão realizadas as avaliações e comparações considerando aspectos como acurácia, consistência, atualidade, fidelidade às informações e capacidade de citação das fontes.

## Objetivo

O objetivo principal é comparar as duas abordagens em tarefas de resposta a perguntas baseadas em conhecimento verificável, buscando identificar os cenários em que cada abordagem apresenta vantagens, limitações e pontos de falha.

## Projeto

**Título:** Comparação entre RAG e LLM-Wiki para Construção e Consulta de Bases de Conhecimento Verificáveis com Grandes Modelos de Linguagem

**Aluno:** Pedro Souza Goularte

**Orientador:** Prof. Dr. Pedro Henrique Lopes Silva

**Departamento:** DECOM/UFOP

**Instituição:** Universidade Federal de Ouro Preto (UFOP)
