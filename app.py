import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Title
st.title("📄 AI Document Assistant")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k":10})

llm = Ollama(model="gemma:2b")

# User input
question = st.text_input("Ask a question from the document:")

if question:

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using only the provided context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response)