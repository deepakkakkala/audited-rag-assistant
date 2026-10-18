import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "./chroma_db"
COLLECTION_NAME = "audited_knowledge_base"

def run_ingestion():
    files = glob.glob("./data/*.md") + glob.glob("./data/*.txt")
    if not files:
        print("No documents found in ./data/. Place markdown or text files there.")
        return

    documents = []
    for file_path in files:
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

    # Chunking: 500 characters with 50-character overlap for context continuity
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    # Initialize Chroma Persistent Client
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    # Using Chroma's default sentence-transformers model (runs locally, zero cost)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()

    # Recreate collection to prevent duplicate inserts during testing
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [doc.page_content for doc in chunks]
    metadatas = [
        {
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "start_index": doc.metadata.get("start_index", 0)
        }
        for doc in chunks
    ]

    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Ingestion complete. {len(ids)} chunks indexed in '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    run_ingestion()