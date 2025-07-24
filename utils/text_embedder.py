import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def embed_chunks(chunks):
    embeddings = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            embeddings.append(np.zeros(768))  # fallback for empty
            continue

        chunk = chunk[:3000]  # Limit size for API

        try:
            res = genai.embed_content(
                model="models/embedding-001",
                content=chunk,
                task_type="retrieval_document"
            )
            embeddings.append(res["embedding"])
        except Exception as e:
            print(f"Embedding failed for chunk {i}: {e}")
            embeddings.append(np.zeros(768))  # fallback

    return np.array(embeddings)

def embed_query(query):
    if not query.strip():
        return np.zeros(768)
    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_document"
        )
        return res["embedding"]
    except Exception as e:
        print(f"Embedding failed for query: {e}")
        return np.zeros(768)

def retrive_relevent_chunks(query, chunks, chunk_embeddings, top_k=5):
    query_embed = embed_query(query)

    # Catch zero vector query
    if np.all(query_embed == 0):
        raise ValueError("Query embedding is a zero vector. Provide a meaningful query.")

    # Convert to 2D shape for cosine_similarity
    query_embed = np.array(query_embed).reshape(1, -1)
    chunk_embeddings = np.array(chunk_embeddings)

    similarities = cosine_similarity(query_embed, chunk_embeddings)[0]

    # ✅ FIXED: Correct usage of argsort
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    return [chunks[i] for i in top_k_indices]
