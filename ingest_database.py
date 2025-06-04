# FIRST CELL: Set the API key securely
import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyBTPMMwBX5SqoEB3XuvVED1m4vb8bxVK3A"  # Paste your real key here

# NOW import everything else
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4
import shutil

# Clear previous DB if needed
CHROMA_PATH = "chroma_db"
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Setup paths
DATA_PATH = "data"

# Initialize embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Setup Chroma
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Load PDFs
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(raw_documents)

# Generate IDs
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Embed and store
vector_store.add_documents(documents=chunks, ids=uuids)

print("✅ Embeddings created and stored in Chroma DB.")