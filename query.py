from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from sentence_transformers import CrossEncoder


# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database
vector_db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)


# Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 20})


# Reranker model
reranker = CrossEncoder("BAAI/bge-reranker-base")


# Local LLM
llm = Ollama(model="gemma:2b")


while True:

    question = input("\nAsk question (type exit to quit): ")

    if question.lower() == "exit":
        break

    # Step 1: Retrieve chunks
    docs = retriever.invoke(question)

    # Step 2: Prepare pairs for reranking
    pairs = []
    for doc in docs:
        pairs.append((question, doc.page_content))

    # Step 3: Score with reranker
    scores = reranker.predict(pairs)

    # Step 4: Combine scores with docs
    scored_docs = list(zip(scores, docs))

    # Step 5: Sort by score
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Step 6: Take top chunks
    top_docs = [doc for score, doc in scored_docs[:12]]

    print("\nTop chunks after reranking:\n")

    for score, doc in scored_docs[:5]:
        print("Score:", score)
        print(doc.page_content[:200])
        print("------")

    # Step 7: Build context
    context = ""
    print("\nRetrieved Context:\n")

    for i, doc in enumerate(top_docs):
        #print(f"Chunk {i+1}:\n{doc.page_content}\n")
        context += doc.page_content + "\n"


    # Step 8: Prompt
    prompt = f"""
You are an assistant answering questions from a document.

Only answer using the provided context.

If the document lists multiple points, return ALL points from the document.

Do not summarize. Return answer that is faithful to the document.
Return them as numbered points if possible.

You are answering questions from a document.

If the document contains a list, you MUST extract the FULL list.

Do not summarize.
Do not reduce the number of items.
Copy every item exactly from the document.

If the answer is not in the context say:
"I cannot find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""


    # Step 9: LLM generation
    response = llm.invoke(prompt)

    print("\nFinal Answer:\n")
    print(response)