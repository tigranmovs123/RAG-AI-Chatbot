#pip install langchain huggingface_hub faiss-cpu PyPDF2 tiktoken #installing  packages with huggingface api
from PyPDF2 import PdfReader
import os

# Folder where you uploaded your PDF
folder = "Data"

def load_pdfs(folder_path):
    all_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            pdf = PdfReader(path)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            all_texts.append(text)
    return all_texts

# Load the PDF
texts = load_pdfs(folder)
print(f"Loaded {len(texts)} document(s).")
print("Preview of first document:")
#print(texts[0][:500])  # first 500 characters


from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_texts(texts, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

chunks = split_texts(texts)
print(f"Total chunks created: {len(chunks)}")
print(chunks[0])

#!pip install -U langchain-community
#pip install sentence-transformers
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Embed your chunks
vectors = [embeddings.embed_query(chunk) for chunk in chunks]

#print(f"Created {len(vectors)} embeddings.")
#print("First embedding vector preview:")
#print(vectors[0][:10])  # show first 10 number

#storing embeddings
import faiss
import pickle
import numpy as np

numpy_vector = np.array(vectors, dtype=np.float32)
dim = numpy_vector.shape[1]
#print(dim)

#  Create FAISS index using Euclidean distance
index = faiss.IndexFlatL2(dim)  # L2 = Euclidean distance

# Add vectors to the index
index.add(numpy_vector)

print(f"FAISS index created with {index.ntotal} vectors.")

# Chunks.pkl for saving chunks, and doc_index for storing embeddings in faiss
faiss.write_index(index, "doc_index.faiss")

# Save text chunks with pickle
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index and chunks saved!")