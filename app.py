from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from sentence_transformers import CrossEncoder

app = FastAPI()

# Load models (only once)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)

retriever = vector_db.as_retriever(search_kwargs={"k": 20})

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

llm = Ollama(model="gemma:2b")


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
def query_rag(req: QueryRequest):

    docs = retriever.invoke(req.question)

    pairs = [(req.question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored_docs[:8]]

    context = "\n".join([doc.page_content for doc in top_docs])

    prompt = f"""
Answer only from context.

Context:
{context}

Question:
{req.question}
"""

    response = llm.invoke(prompt)

    return {"answer": response}