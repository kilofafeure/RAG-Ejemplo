from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DOCUMENT_PATH = "document.txt"

def load_chunks():
    """Carga el documento y lo divide en chunks."""
    loader = TextLoader(DOCUMENT_PATH, encoding="utf-8")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    return splits

def create_vectorstore(docs=None):
    """
    Crea o carga la base de datos vectorial.
    Si se pasan documentos, los ingesta. Si no, carga la existente.
    """
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if docs:
        # Si hay documentos, creamos/new vectorstore
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        print(f"Ingesta completada. {len(docs)} chunks guardados.")
        return vectorstore
    else:
        # Si no hay documentos, cargamos lo que ya existe en disco
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        count = vectorstore._collection.count()
        print(f"Base de datos cargada. {count} chunks encontrados.")
        return vectorstore