import logging
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

def create_vector_stores(text_documents, image_metadata, embeddings):
    # Text vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
    text_chunks = text_splitter.split_text("\n".join(text_documents))
    if not text_chunks:
        logger.warning("No text chunks created")
        return None, None, None
    
    text_vector_store = FAISS.from_texts(text_chunks, embeddings)
    
    # Image description vector store
    image_descriptions = [img["description"] for img in image_metadata]
    image_vector_store = FAISS.from_texts(image_descriptions, embeddings) if image_descriptions else None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text_vector_store, image_vector_store, text_chunks

def find_relevant_images(question, image_vector_store, image_metadata, embeddings, top_k=1, similarity_threshold=0.7):
    if not image_metadata or not image_vector_store:
        logger.info("No images or VLM disabled")
        return []
    
    question_embedding = embeddings.embed_query(question)
    docs = image_vector_store.similarity_search_by_vector(question_embedding, k=top_k)
    relevant_images = []
    for doc in docs:
        for img in image_metadata:
            if img["description"] == doc.page_content and image_vector_store.similarity_search_by_vector(embeddings.embed_query(img["description"]), k=1)[0].page_content == doc.page_content:
                similarity = np.dot(question_embedding, embeddings.embed_query(img["description"])) / (
                    np.linalg.norm(question_embedding) * np.linalg.norm(embeddings.embed_query(img["description"]))
                )
                if similarity >= similarity_threshold:
                    relevant_images.append({"path": img["path"], "type": img["type"], "description": img["description"], "similarity": similarity})
    
    relevant_images = sorted(relevant_images, key=lambda x: x["similarity"], reverse=True)[:top_k]
    logger.debug(f"Found {len(relevant_images)} relevant images")
    return relevant_images