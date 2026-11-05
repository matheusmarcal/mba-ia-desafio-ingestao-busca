# Ingestão e Busca Semântica com LangChain e Postgres

Solução do [desafio de Ingestão e Busca](https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca) do MBA em IA da Full Cycle.

O projeto lê um PDF, guarda os trechos como vetores no PostgreSQL com pgVector e responde perguntas pelo terminal usando **somente** o conteúdo do PDF.

```
PERGUNTA: Qual o faturamento da empresa SuperTechIABrazil?
RESPOSTA: R$ 10.000.000,00

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Stack

- Python 3.12 ou 3.13 (as versões fixadas no `requirements.txt` ainda não têm pacotes para o 3.14)
- LangChain
- Google Gemini
  - embeddings: `models/gemini-embedding-001` (3072 dimensões)
  - LLM: `gemini-3.5-flash-lite`
- PostgreSQL 17 + pgVector via Docker Compose

## Como funciona

**Ingestão (`src/ingest.py`)**
1. Carrega o PDF com `PyPDFLoader`.
2. Divide o texto em chunks de 1000 caracteres com overlap de 150 (`RecursiveCharacterTextSplitter`).
3. Gera o embedding de cada chunk e grava no pgVector (`PGVector`).

A collection é recriada a cada execução, então rodar a ingestão de novo não duplica dados. Os chunks são enviados em lotes de 20. Se a cota gratuita do Gemini estourar, o script espera 60s e tenta de novo.

**Busca (`src/search.py`)**
1. Vetoriza a pergunta.
2. Busca os 10 chunks mais relevantes com `similarity_search_with_score(k=10)`.
3. Monta o prompt do desafio com esses trechos.
4. Chama a LLM e devolve a resposta.

**Chat (`src/chat.py`)**
- CLI interativa. Digite `sair` (ou Ctrl+D) para encerrar.

## Pré-requisitos

- Docker e Docker Compose
- Python 3.12 ou 3.13
- Uma API key do Google: https://aistudio.google.com/apikey

## Execução

### 1. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Preencha `GOOGLE_API_KEY` no `.env`. Os demais valores já vêm com padrões funcionais.

| Variável | Descrição |
| --- | --- |
| `GOOGLE_API_KEY` | Chave da API do Google |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings (padrão `models/gemini-embedding-001`) |
| `GOOGLE_LLM_MODEL` | Modelo de chat (padrão `gemini-3.5-flash-lite`) |
| `DATABASE_URL` | Conexão com o Postgres (padrão `postgresql+psycopg://postgres:postgres@localhost:5433/rag`) |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector (padrão `documento_pdf`) |
| `PDF_PATH` | PDF a ingerir, relativo à raiz ou absoluto (padrão `document.pdf`) |

> O Postgres do Docker é exposto na porta **5433** do host. Assim ele não conflita com um Postgres local que já esteja usando a 5432.

### 2. Crie o ambiente virtual e instale as dependências

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Suba o banco de dados

```bash
docker compose up -d
```

### 4. Execute a ingestão do PDF

```bash
python src/ingest.py
```

### 5. Rode o chat

```bash
python src/chat.py
```

## Trocando o modelo de embeddings

A tabela de vetores é criada com a dimensão do modelo usado na primeira ingestão. Se você trocar `GOOGLE_EMBEDDING_MODEL` por um modelo de dimensão diferente, apague o volume do banco e refaça a ingestão:

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```
