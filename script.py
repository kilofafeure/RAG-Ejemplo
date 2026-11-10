from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------- PARTE 1: Ingesta ----------
loader = TextLoader("document.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"{len(splits)} chunks creados")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# ---------- PARTE 2: Consulta RAG ----------
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