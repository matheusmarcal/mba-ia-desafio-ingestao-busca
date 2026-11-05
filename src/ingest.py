import os
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import PROJECT_ROOT, build_vector_store, require_env

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# A camada gratuita do Gemini limita os tokens de embedding por minuto.
# Enviamos os chunks em lotes e, se a cota estourar, esperamos e tentamos de novo.
BATCH_SIZE = 20
QUOTA_WAIT_SECONDS = 60
MAX_RETRIES = 5


def resolve_pdf_path():
    (pdf_path,) = require_env("PDF_PATH")
    path = Path(pdf_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {path}")
    return path


def is_quota_error(error):
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower()


def add_in_batches(store, chunks):
    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        ids = [f"chunk-{index}" for index in range(start, start + len(batch))]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                store.add_documents(batch, ids=ids)
                break
            except Exception as error:
                if not is_quota_error(error) or attempt == MAX_RETRIES:
                    raise
                print(f"  Limite de cota da API atingido; aguardando {QUOTA_WAIT_SECONDS}s "
                      f"(tentativa {attempt}/{MAX_RETRIES})...")
                time.sleep(QUOTA_WAIT_SECONDS)
        print(f"  {min(start + len(batch), total)}/{total} chunks salvos")


def ingest_pdf():
    pdf_path = resolve_pdf_path()
    print(f"Lendo PDF: {pdf_path}")
    pages = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise RuntimeError("Nenhum texto extraído do PDF.")
    print(f"{len(pages)} página(s) divididas em {len(chunks)} chunks.")

    store = build_vector_store()
    # Recria a collection para que rodar a ingestão de novo não duplique os chunks.
    store.delete_collection()
    store.create_collection()

    add_in_batches(store, chunks)
    collection = os.getenv("PG_VECTOR_COLLECTION_NAME")
    print(f"Ingestão concluída: {len(chunks)} chunks salvos na collection '{collection}'.")


if __name__ == "__main__":
    ingest_pdf()
