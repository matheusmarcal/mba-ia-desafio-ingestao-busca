import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def require_env(*names):
    """Lê variáveis obrigatórias do .env, falhando com mensagem clara se faltar alguma."""
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            f"Variáveis de ambiente ausentes: {', '.join(missing)}. "
            "Copie .env.example para .env e preencha os valores."
        )
    return [os.getenv(name) for name in names]


def build_vector_store():
    """PGVector configurado a partir do .env. Usado tanto na ingestão quanto na busca,
    garantindo que os dois lados usem o mesmo modelo de embeddings e a mesma collection."""
    require_env("GOOGLE_API_KEY")
    embedding_model, database_url, collection = require_env(
        "GOOGLE_EMBEDDING_MODEL", "DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"
    )
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, transport="rest")
    return PGVector(
        embeddings=embeddings,
        collection_name=collection,
        connection=database_url,
        use_jsonb=True,
    )


def format_context(results):
    """Concatena o conteúdo dos chunks retornados por similarity_search_with_score."""
    return "\n\n".join(document.page_content for document, _score in results)


def search_prompt(question=None):
    """Monta a chain RAG: pergunta -> busca k=10 no pgVector -> prompt -> LLM -> texto.

    Sem argumento, retorna a chain (ou None se a inicialização falhar).
    Com `question`, executa a chain e retorna a resposta.
    """
    try:
        (llm_model,) = require_env("GOOGLE_LLM_MODEL")
        store = build_vector_store()
        llm = ChatGoogleGenerativeAI(model=llm_model, transport="rest")
    except Exception as error:
        print(f"Erro ao inicializar a busca: {error}")
        return None

    def retrieve_context(pergunta):
        results = store.similarity_search_with_score(pergunta, k=TOP_K)
        return format_context(results)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = (
        {"contexto": RunnableLambda(retrieve_context), "pergunta": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    if question is not None:
        return chain.invoke(question)
    return chain
