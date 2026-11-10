from vectorstore import create_vectorstore
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Cargar la base de datos ya existente
vectorstore = create_vectorstore() # Sin documentos, carga desde disco

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

# Aquí puedes cambiar la pregunta fácilmente
query = "¿Cuál es la última tarea de la lista?"
result = rag_chain.invoke(query)

print(result)