import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Configurações
PASTA_BASE = "base"
PASTA_DB = "db"

def verificar_banco_vetorial():
    if not os.path.exists(PASTA_DB):
        st.error("Banco de dados não encontrado!")
        st.info("Execute primeiro o script `criar_db.py` para processar os PDFs.")
        return False
    return True

def main():
    # Configuração da página
    st.set_page_config(
        page_title="Assistente de PDFs em Português",
        
    )
    st.title("RAG com Groq + LangChain")

    # Verificação inicial
    load_dotenv()
    if not verificar_banco_vetorial():
        st.stop()

    # Carrega o banco vetorial
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=PASTA_DB, embedding_function=embeddings)

    # Configuração do modelo (CORREÇÃO DO ERRO)
    modelo_selecionado = st.selectbox(
        "Selecione o modelo:",
        ["llama3-70b-8192", "mixtral-7b-32768"],
        index=0
    )

    # Prompt personalizado em português
    template = """Você é um assistente que responde em português brasileiro. Use o contexto abaixo:

    {context}

    Pergunta: {question}
    Resposta em português brasileiro:"""

    QA_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Configura o LLM (VERSÃO CORRIGIDA)
    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=modelo_selecionado,
            temperature=0.1
        )
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        st.stop()

    # Cadeia de QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # Campo de consulta
    query = st.text_input("Faça sua pergunta em português:")

    if query:
        with st.spinner("Processando..."):
            try:
                resposta = qa_chain.invoke({"query": query})
                
                st.subheader("Resposta")
                st.write(resposta["result"])

                with st.expander("Fontes consultadas"):
                    for i, doc in enumerate(resposta["source_documents"]):
                        st.write(f"{os.path.basename(doc.metadata['source'])} (Página {doc.metadata['page'] + 1})")
                        st.write(doc.page_content)
                        st.divider()

            except Exception as e:
                st.error(f"Erro: {str(e)}")

if __name__ == "__main__":
    main()