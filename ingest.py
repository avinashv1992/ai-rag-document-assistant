from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("data\RS Essentials (MultiCAD Edition).pdf")
documents = loader.load()

print("Loaded pages:", len(documents))

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

#Filtering out tiny chunks that are not useful for retrieval
filtered_chunks = []

for chunk in chunks:
    text = chunk.page_content.strip()

    if len(text) > 200:   # remove tiny useless chunks
        filtered_chunks.append(chunk)

chunks = filtered_chunks

clean_chunks = []

for chunk in chunks:
    text = chunk.page_content.lower()

    if (
        len(text) > 150
        and "discord" not in text
        and "chapter" not in text
        and "score:" not in text
    ):
        clean_chunks.append(chunk)

chunks = clean_chunks

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector database
vector_db = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="vector_db"
)

vector_db.persist()

print("Vector database created")