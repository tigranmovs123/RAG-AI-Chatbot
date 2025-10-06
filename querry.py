import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
# Load FAISS index
index = faiss.read_index("doc_index.faiss")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

client = Groq(api_key="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")
def retrieve_chunks(query_text, top_k=5):
    query_vector = model.encode([query_text])
    distances, indices = index.search(query_vector, top_k)

    # Access chunks directly since they are strings
    results = [chunks[idx] for idx in indices[0]]

    return results


def answer_query(query_text):
    relevant_chunks = retrieve_chunks(query_text)
    context = "\n\n".join(relevant_chunks)
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query_text}\nAnswer:"

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content.strip()

# Simple console test
if __name__ == "__main__":
    while True:
        user_input = input("Enter your question (or 'exit'): ")
        if user_input.lower() == "exit":
            break
        answer = answer_query(user_input)
        print(f"\nAnswer: {answer}\n")
