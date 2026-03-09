### RAGGIFY

RAG Semantic Search (Doc Lake + Qdrant + Sentence Transformers)

A simple Retrieval-Augmented Generation (RAG) pipeline that performs semantic search over .txt/.pdf documents stored in a local doc lake directory. It uses Sentence Transformers for encoding text into embeddings and Qdrant as the vector database for similarity retrieval.

Drop .txt files into '/doc_lake', ingest them, then query the collection to retrieve the most relevant passages. Vector store is cleared and reloaded on each application run.

