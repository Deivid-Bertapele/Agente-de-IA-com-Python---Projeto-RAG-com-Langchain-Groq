import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configurações
PASTA_BASE = "base"
PASTA_DB = "db"

def carregar_pdfs():
    """Carrega todos os PDFs da pasta 'base'."""
    if not os.path.exists(PASTA_BASE):
        print(f"Erro: Pasta '{PASTA_BASE}' não encontrada!")
        return None

    loader = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    return loader.load()

def processar_documentos():
    """Divide os textos em pedaços (chunks)."""
    documentos = carregar_pdfs()
    if not documentos:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_documents(documentos)

def criar_banco_vetorial():
    """Cria o ChromaDB com os embeddings."""
    chunks = processar_documentos()
    if not chunks:
        print("Nenhum documento para processar.")
        return False

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PASTA_DB
    )
    print(f"Banco vetorial criado em '{PASTA_DB}'")
    return True

if __name__ == "__main__":
    load_dotenv()
    criar_banco_vetorial()