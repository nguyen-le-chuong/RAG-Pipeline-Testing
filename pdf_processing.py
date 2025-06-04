import os
import logging
import numpy as np
from yomitoku.data.functions import load_pdf
from PIL import Image
from yomitoku.document_analyzer import DocumentAnalyzer
from pathlib import Path
from utils import detect_encoding, get_file_hash
import torch
logger = logging.getLogger(__name__)

def extract_text_from_pdf(args, pdf_path, analyzer, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    pdf_hash = get_file_hash(pdf_path)
    cache_file = os.path.join(cache_dir, f"{os.path.basename(pdf_path)}_{pdf_hash}.md")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            logger.debug(f"Loading cached text for {pdf_path}")
            return f.read()
    
    # Convert PDF to images
    pdf_path = Path(pdf_path)
    images = load_pdf(pdf_path)
    text = ""
    
    # Initialize YomiToku if not provided
    if analyzer is None:
        logger.warning("No YomiToku analyzer provided, initializing default")
        analyzer = DocumentAnalyzer(visualize=False, device="cuda" if torch.cuda.is_available() else "cpu")
    
    for i, image in enumerate(images):
        # Process image with YomiToku
        results, _, _ = analyzer(image)
        dirname = pdf_path.parent.name
        filename = pdf_path.stem

        # cv2.imwrite(
        #    os.path.join(args.outdir, f"{dirname}_{filename}_p{page+1}.jpg"), img
        # )

        out_path = os.path.join(
            args.output_folder, f"{dirname}_{filename}/p{i + 1}.md"
        )
        page_text = results.to_markdown(
                out_path,
                ignore_line_break=True,
                img=image,
                export_figure=True,
                export_figure_letter=True,
                figure_width=100,
                figure_dir="figures",
                encoding="utf-8",
            )
        text += f"--- Page {i+1} ---\n{page_text}\n"
    
    # Cache the extracted text
    with open(cache_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    logger.debug(f"Extracted text from {pdf_path}, {len(text)} chars")
    return text

def extract_images_from_pdf(args, pdf_path, output_folder):
    pdf_hash = get_file_hash(pdf_path)
    pdf_path = Path(pdf_path)
    image_paths = []
    dirname = pdf_path.parent.name
    filename = pdf_path.stem
    # print(args.output_folder)
    output_folder = os.path.join(args.output_folder, dirname + "_" + filename + "/figures")
    images = os.listdir(output_folder)
    # print(images)
    for i, image in enumerate(images):
        image_path = output_folder + "/" + image
        if not os.path.exists(image_path):
            image.save(image_path, "PNG")
        image_paths.append(image_path)
    
    logger.debug(f"Extracted {len(image_paths)} images from {pdf_path}")
    return image_paths