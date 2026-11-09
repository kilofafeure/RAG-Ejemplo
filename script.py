from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Parte 1 - Ingesta y preparación de datos (rag.png)

# 1. - Carga documento
loader = TextLoader("document.txt", encoding="utf-8")
docs = loader.load()

#2. - Dividir en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
splits = text_splitter.split_documents(docs)

#3. - Embeddings con Ollama
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

#4. Vector DB (Chroma)
vectorstore = Chroma.from_documents(documents = splits, embbeding = embeddings)

# Parte 2 - Consulta y generación (rag.png)