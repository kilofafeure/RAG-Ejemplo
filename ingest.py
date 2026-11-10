from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma_db"

# ---------- PARTE 1: Ingesta ----------
loader = TextLoader("document.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"{len(splits)} chunks creados")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Si la carpeta chroma_db ya existe con datos viejos, delete_if_true la borra
# para empezar de cero. Si quieres actualizar sin borrar, gestiona esto con cuidado.
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
)

print(f"Ingesta completada. {len(splits)} chunks guardados en {CHROMA_DIR}")