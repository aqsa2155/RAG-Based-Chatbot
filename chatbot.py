# chatbot.py

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv
import os

# Load environment variables from .env (if running locally)
load_dotenv()

# Or set API key directly (use this in Colab)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCNFgrjfjWueEC0zKMIPnH2ZmEhkfdO9RU"  # Replace with your actual key

# Configuration
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# Initialize embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

# Load Chroma vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Set retriever
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# Stream response for each user message
def stream_response(message, history):
    docs = retriever.invoke(message)
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
You are a helpful assistant. Use only the information in the "Knowledge" section to answer the user's question.

Question: {message}

Conversation history: {history}

Knowledge: {knowledge}
"""

    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response.content
        yield partial_message

# Create Gradio chatbot interface
chatbot = gr.ChatInterface(
    fn=stream_response,
    textbox=gr.Textbox(
        placeholder="Ask a question about your documents...",
        container=False,
        autoscroll=True,
        scale=7
    ),
)

# Launch the chatbot
chatbot.launch(share=True)