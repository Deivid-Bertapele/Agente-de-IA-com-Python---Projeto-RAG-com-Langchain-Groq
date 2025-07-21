# Agente-de-IA-com-Python---Projeto-RAG-com-Langchain-Groq

# 📚 Assistente de Consulta a PDFs com IA

Um sistema RAG (Retrieval-Augmented Generation) que permite consultar documentos PDF em português usando modelos de linguagem avançados.

## 🔍 Funcionalidades

- **Upload e processamento automático** de PDFs
- **Busca semântica** nos documentos
- **Respostas em português** com referências às páginas originais
- Suporte a múltiplos modelos (**Llama 3, Mixtral, Gemma**)
- Interface web intuitiva com **Streamlit**

## 🛠️ Tecnologias Utilizadas

| Tecnologia       | Descrição                           |
|------------------|-----------------------------------|
| Python           | Linguagem principal               |
| LangChain        | Framework para aplicações com IA   |
| ChromaDB         | Banco de dados vetorial            |
| Hugging Face     | Modelos de embeddings              |
| Groq API         | Infraestrutura para LLMs           |
| Streamlit        | Interface web                      |

## ⚙️ Como Executar

### Pré-requisitos
- Python 3.10+
- Conta na [Groq Cloud] (https://console.groq.com/) (para API key)
- PDFs para análise (coloque na pasta `base`)

### Instalação

1. Clone o repositório:
```bash
git clone do arquivo
cd do arquivo
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Crie um arquivo .env com sua API key:
```bash
GROQ_API_KEY=sua_chave_aqui
```

## Uso

1. Processe seus PDFs:
```bash
python criar_db.py
```

2. Inicie a interface web:
```bash
streamlit run app.py
```
  

## 📌 Melhorias Futuras:

Adicionar suporte a outros formatos (DOCX, PPTX)

Implementar histórico de conversas

Adicionar autenticação de usuários

Criar versão dockerizada


