This project implements a Retrieval-Augmented Generation pipeline that allows a Large Language Model (LLM) to answer questions based on custom PDF documents.
It loads and splits PDFs, generates semantic embeddings, stores them in a FAISS vector database, and performs contextual retrieval before sending the query to the LLM.
The result is a chatbot that gives is context-specific answers instead of generic responses.
