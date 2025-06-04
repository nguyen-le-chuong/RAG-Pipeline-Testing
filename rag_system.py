import logging
import pandas as pd
import torch
import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pdf_processing import extract_text_from_pdf, extract_images_from_pdf
from image_processing import process_image
from vector_stores import create_vector_stores, find_relevant_images
from utils import detect_encoding
from prompts import prompt_with_options
from PIL import Image

logger = logging.getLogger(__name__)

def function_calling(query, relevant_images, vision_model, vision_processor, disable_vlm=False):
    if disable_vlm or not relevant_images:
        logger.info("VLM disabled or no relevant images")
        return None
    
    image = Image.open(relevant_images[0]["path"]).convert("RGB")
    image_type = relevant_images[0]["type"].lower()
    base_description = relevant_images[0]["description"]
    
    if "diagram" in query.lower() and "diagram" in image_type:
        desc_prompt = f"Using the base description: '{base_description}', analyze this diagram further in the context of the question: '{query}'. Provide a detailed response focusing on components relevant to the question."
    elif "table" in query.lower() and "table" in image_type:
        desc_prompt = f"Using the base description: '{base_description}', extract additional details from this table relevant to the question: '{query}'."
    elif "photo" in image_type:
        desc_prompt = f"Using the base description: '{base_description}', describe this photo further in the context of the question: '{query}'."
    elif "graph" in image_type or "chart" in image_type:
        desc_prompt = f"Using the base description: '{base_description}', analyze this graph/chart further for the question: '{query}'."
    else:
        desc_prompt = f"Using the base description: '{base_description}', provide additional details for this image relevant to the question: '{query}'."
    
    inputs = vision_processor(text=desc_prompt, images=[image], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = vision_model.generate(**inputs, max_new_tokens=300)
    enhanced_description = vision_processor.decode(outputs[0], skip_special_tokens=True).replace(desc_prompt, "").strip()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return enhanced_description

# def create_rag_chain(text_vector_store, llm, top_k=3):
#     if not text_vector_store:
#         logger.error("Text vector store is None")
#         return None
#     retriever = text_vector_store.as_retriever(search_kwargs={"k": top_k})
    
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={"prompt": prompt_with_options},
#         input_key="query"
#     )
#     return qa_chain

def run_rag_system(args, pdf_path, question, options, output_folder, vision_model, vision_processor, llm, embeddings, top_k=3, similarity_threshold=0.7, analyzer=None):
    if not question:
        logger.error(f"No question for PDF {pdf_path}")
        return "Error: No question provided"
    
    documents, image_metadata = process_documents(args, pdf_path, output_folder, vision_model, vision_processor, analyzer)
    if not documents and not image_metadata:
        logger.error(f"No data processed for {pdf_path}")
        return "No data processed."
    
    text_vector_store, image_vector_store, text_chunks = create_vector_stores(documents, image_metadata, embeddings)
    if not text_vector_store:
        logger.error("Failed to create text vector store")
        return "Failed to create vector store."
    
    relevant_images = find_relevant_images(question, image_vector_store, image_metadata, embeddings, top_k, similarity_threshold)
    function_result = function_calling(question, relevant_images, vision_model, vision_processor, disable_vlm=args.disable_vlm)
    function_results = function_result if function_result else "なし"
    
    retriever = text_vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.get_relevant_documents(question)
    context_text = "\n".join([doc.page_content for doc in relevant_docs]) or "No context found."
    
    inputs = {
        "question": question,
        "options": options,
        "context": context_text,
        "function_results": function_results
    }
    formatted_prompt = prompt_with_options.format(**inputs)
    response = llm.invoke(formatted_prompt)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return response

def process_documents(args, pdf_path, output_folder, vision_model, vision_processor, analyzer):
    documents = []
    image_metadata = []
    
    # Extract text using YomiToku
    text = extract_text_from_pdf(args, pdf_path, analyzer)
    if text:
        documents.append(text)
    
    # Extract and process images
    image_paths = extract_images_from_pdf(args, pdf_path, output_folder)
    for image_path in image_paths:
        image_type, description = process_image(image_path, vision_model, vision_processor)
        if image_type and description:
            image_metadata.append({"path": image_path, "type": image_type, "description": description})
    
    logger.debug(f"Processed: {len(documents)} text chunks, {len(image_metadata)} images")
    return documents, image_metadata

def process_csv_questions(args,csv_path, pdf_folder, output_folder, llm, vision_model, vision_processor, embeddings, top_k=3, similarity_threshold=0.7, analyzer=None):

    encoding = detect_encoding(csv_path)
    df = pd.read_csv(csv_path, encoding=encoding)
    required_columns = ['id', 'pdf_name', 'question']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"CSV missing columns: {required_columns}")
        return [{"id": "N/A", "answer": "Error: CSV missing columns"}]
    
    results = []
    for _, row in df.iterrows():
        question_id = row['id']
        pdf_name = row['pdf_name']
        question = row['question']
        
        options = []
        for i in range(1, 11):
            option_col = f'option_{i}'
            if option_col in df.columns and not pd.isna(row[option_col]):
                option_value = str(row[option_col]).strip()
                if option_value:
                    options.append(option_value)
        
        options_str = " ".join([f"{chr(65+i)}) {opt}" for i, opt in enumerate(options)]) if options else "No options."
        
        pdf_path = os.path.join(pdf_folder, pdf_name)
        
        answer = run_rag_system(args, pdf_path, question, options_str, output_folder, vision_model, vision_processor, llm, embeddings, top_k, similarity_threshold, analyzer)
        results.append({"id": question, "answer": answer})
        print(f"Processed question {question}: \n Answer: {answer}")
    
    return results