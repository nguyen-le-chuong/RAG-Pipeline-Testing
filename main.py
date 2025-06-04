import os
import argparse
import torch
from yomitoku.document_analyzer import DocumentAnalyzer
from logging_setup import setup_logging
from models import initialize_llm, initialize_vlm, initialize_embeddings
from rag_system import process_csv_questions

def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimized RAG system with CoT and YomiToku OCR")
    parser.add_argument("--disable-vlm", action="store_true", help="Disable vision-language model")
    parser.add_argument("--csv-path", default="/mnt/mmlab2024nas/huycq/huycq/Chatbot/datasets/train.csv", help="Path to CSV file")
    parser.add_argument("--pdf-folder", default="/mnt/mmlab2024nas/huycq/huycq/Chatbot/datasets/train_pdf_files", help="Folder with PDF files")
    parser.add_argument("--output_folder", default="output", help="Folder for extracted images")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top chunks to retrieve")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Minimum similarity score for images")
    return parser.parse_args()

def main():
    args = parse_arguments()
    setup_environment()
    logger = setup_logging(args.log_level)
    os.makedirs(args.output_folder, exist_ok=True)

    analyzer = DocumentAnalyzer(visualize=False, device="cuda" if torch.cuda.is_available() else "cpu")


    llm= initialize_llm(args)
    vision_model, vision_processor = initialize_vlm(args)
    embeddings = initialize_embeddings()


    results = process_csv_questions(\
        args=args,
        csv_path=args.csv_path,
        pdf_folder=args.pdf_folder,
        output_folder=args.output_folder,
        llm=llm,
        vision_model=vision_model,
        vision_processor=vision_processor,
        embeddings=embeddings,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        analyzer=analyzer
    )
    

if __name__ == "__main__":
    main()