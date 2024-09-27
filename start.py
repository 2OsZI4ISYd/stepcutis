import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'projectpackages'))
sys.path.append(os.path.join(project_root, 'projectpackages', 'LakeOCR'))
import argparse
import fitz
import subprocess
import shutil
from process_document import process_pdf
import torch
from projectpackages.CRAFT import CRAFTModel
from projectpackages.surya.model.detection.model import load_model, load_processor
from projectpackages.surya.settings import settings
from projectpackages.surya.model.ordering.processor import load_processor as load_ordering_processor
from projectpackages.surya.model.ordering.model import load_model as load_ordering_model
import traceback
import logging
from bs4 import BeautifulSoup
from transformers import AutoTokenizer
from projectpackages.LakeOCR.GOT.model import GOTQwenForCausalLM
from projectpackages.LakeOCR.GOT.model.plug.blip_process import BlipImageEvalProcessor
from projectpackages.LakeOCR.GOT.utils.utils import disable_torch_init

def ensure_folders_exist():
    folders = ['partitions', 'regionimages', 'shelves', 'weights']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def delete_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def delete_shelves_contents():
    shelves_dir = os.path.join(os.getcwd(), 'shelves')
    for item in os.listdir(shelves_dir):
        item_path = os.path.join(shelves_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def split_pdf(input_pdf, chunk_size):
    doc = fitz.open(input_pdf)
    num_pages = doc.page_count
    num_chunks = (num_pages + chunk_size - 1) // chunk_size

    num_digits = len(str(num_chunks))

    pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]
    shelves_dir = os.path.join(os.getcwd(), 'shelves')
    os.makedirs(shelves_dir, exist_ok=True)
    chunk_input_dir = os.path.join(shelves_dir, f'chunk_input_{pdf_name}')
    os.makedirs(chunk_input_dir, exist_ok=True)

    for i in range(num_chunks):
        start_page = i * chunk_size
        end_page = min((i + 1) * chunk_size, num_pages)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
        
        chunk_num = str(i + 1).zfill(num_digits)
        chunk_output = os.path.join(chunk_input_dir, f"chunk_{chunk_num}.pdf")
        
        chunk_doc.save(chunk_output)
        chunk_doc.close()

    doc.close()
    return num_chunks, chunk_size

def process_chunks(chunk_output_dir, output_html, create_single_file=True):
    chunk_files = sorted([f for f in os.listdir(chunk_output_dir) if f.endswith(".html")])
    combined_html = ""

    for chunk_file in chunk_files:
        with open(os.path.join(chunk_output_dir, chunk_file), "r", encoding="utf-8") as f:
            chunk_html = f.read()
            combined_html += chunk_html + "\n"

    if create_single_file:
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(combined_html)
    else:
        return combined_html

def get_dataset_filename(original_pdf_filename, page_num):
    base_filename = os.path.splitext(os.path.basename(original_pdf_filename))[0]
    return f"{base_filename}_{page_num}.html"

def process_dataset_output(combined_html, original_pdf_filename):
    output_dir = os.path.dirname(original_pdf_filename)
    soup = BeautifulSoup(combined_html, 'html.parser')
    pages = soup.find_all('div', class_='page')
    
    for i, page in enumerate(pages, start=1):
        # Remove the page-number div
        page_number_div = page.find('div', class_='page-number')
        if page_number_div:
            page_number_div.decompose()
        
        # Remove all base64-encoded images (pictures and figures)
        for img in page.find_all('img'):
            if img.get('src', '').startswith('data:image'):
                img.decompose()
        
        # Remove empty figure elements
        for figure in page.find_all('figure'):
            if not figure.contents:
                figure.decompose()
        
        page_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{os.path.splitext(os.path.basename(original_pdf_filename))[0]} - Page {i}</title>
        </head>
        <body>
            {str(page)}
        </body>
        </html>
        """
        
        output_filename = get_dataset_filename(original_pdf_filename, i)
        with open(os.path.join(output_dir, output_filename), "w", encoding="utf-8") as f:
            f.write(page_html)

def find_pdf_files(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def initialize_ocr_model():
    disable_torch_init()
    model_name = os.path.join(os.getcwd(), 'GOT-OCR2_0')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
    model.to(device='cuda', dtype=torch.float16)
    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    return tokenizer, model, image_processor, image_processor_high

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing PDF files and subdirectories")
    parser.add_argument("--chunk_size", type=int, default=10, help="Number of pages per chunk (default: 10)")
    parser.add_argument("--dataset", action="store_true", help="Create individual HTML files for each page")
    args = parser.parse_args()

    ensure_folders_exist()
    delete_shelves_contents()

    # Initialize models
    craft_word_model = CRAFTModel('weights/', torch.device('cuda' if torch.cuda.is_available() else 'cpu'), use_refiner=False, fp16=True)
    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()
    order_model = load_ordering_model()
    order_processor = load_ordering_processor()
    
    # Initialize OCR model
    ocr_tokenizer, ocr_model, ocr_image_processor, ocr_image_processor_high = initialize_ocr_model()

    pdf_files = find_pdf_files(args.input_dir)
    for input_pdf in pdf_files:
        output_html = os.path.splitext(input_pdf)[0] + '.html'

        try:
            logging.info(f"Processing file: {input_pdf}")
            
            if not os.path.isfile(input_pdf):
                raise FileNotFoundError(f"Input PDF file not found: {input_pdf}")

            num_chunks, chunk_size = split_pdf(input_pdf, args.chunk_size)

            pdf_name = os.path.splitext(os.path.basename(input_pdf))[0]
            chunk_input_dir = os.path.join(os.getcwd(), 'shelves', f'chunk_input_{pdf_name}')
            chunk_output_dir = os.path.join(os.getcwd(), 'shelves', f'chunk_output_{pdf_name}')
            os.makedirs(chunk_output_dir, exist_ok=True)

            chunk_files = sorted([f for f in os.listdir(chunk_input_dir) if f.endswith(".pdf")])
            for chunk_num, chunk_file in enumerate(chunk_files, start=1):
                chunk_input = os.path.join(chunk_input_dir, chunk_file)
                chunk_output = os.path.join(chunk_output_dir, f"{os.path.splitext(chunk_file)[0]}.html")
                logging.info(f"Processing chunk: {chunk_file}")
                process_pdf(chunk_input, chunk_output, craft_word_model, model, processor, det_model, det_processor, order_model, order_processor, 
                            ocr_tokenizer, ocr_model, ocr_image_processor, ocr_image_processor_high,
                            chunk_num, chunk_size, args.dataset, os.path.dirname(input_pdf), input_pdf)

            if args.dataset:
                combined_html = process_chunks(chunk_output_dir, output_html, create_single_file=False)
                process_dataset_output(combined_html, input_pdf)
            else:
                process_chunks(chunk_output_dir, output_html)

            delete_files_in_directory(chunk_input_dir)
            delete_files_in_directory(chunk_output_dir)
            
            logging.info(f"Successfully processed {input_pdf}")

        except Exception as e:
            logging.error(f"Error processing {input_pdf}:")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error message: {str(e)}")
            logging.error("Traceback:", exc_info=True)
            print(f"Error processing {input_pdf}:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            continue

    delete_shelves_contents()

if __name__ == "__main__":
    main()