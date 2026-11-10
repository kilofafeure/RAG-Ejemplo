from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"

# ---------- PARTE 2: Consulta RAG ----------

# 1. Cargar la base de datos ya existente desde disco
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Importante: usar Chroma(persist_directory=...) en lugar de from_documents()
# Esto reconstruye el vectorstore desde los datos guardados
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(model="llama3:8b", temperature=0)

template = """Responde a la pregunta usando solo la siguiente información:

{context}

Pregunta: {query}"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "query": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "¿Cuál es la última tarea de la lista?"
result = rag_chain.invoke(query)

print(result)