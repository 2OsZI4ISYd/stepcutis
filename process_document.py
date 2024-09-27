import os
import fitz
from tqdm import tqdm
import cv2
import numpy as np
from projectpackages.CRAFT import CRAFTModel
from layout import get_layout
from ocr import process_images
from projectpackages.surya.layout import batch_layout_detection
import subprocess
import math
import shutil

def ensure_folders_exist():
    folders = ['partitions', 'regionimages']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def polygon_to_bbox(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    return x1, y1, x2, y2

def calculate_bbox_height(bbox):
    _, y1, _, y2 = bbox
    return abs(y2 - y1)

def calculate_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return abs(x2 - x1)

def draw_bounding_boxes(image, bboxes, average_height, average_width):
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bbox_height = abs(y2 - y1)
        bbox_width = abs(x2 - x1)
        x_initial = int(x1 + (bbox_height / 4))
        x_final = int(x2 - (bbox_height / 4))
        
        vertical_center = (y1 + y2) / 2
        vertical_1 = int(vertical_center - (bbox_height))
        vertical_4 = int(vertical_center + (bbox_height))

        if x_initial < x_final:
            for x in range(int(x_initial), int(x_final), max(1, math.ceil(bbox_height/4))):
                cv2.putText(image, "O", (x * 2, int(vertical_center)), cv2.FONT_HERSHEY_SIMPLEX, bbox_height / 50, (0, 0, 0), math.ceil(bbox_height / 20))
    return image

def get_dataset_filename(original_pdf_filename, page_num):
    base_filename = os.path.splitext(os.path.basename(original_pdf_filename))[0]
    return f"{base_filename}_{page_num}.png"

def save_dataset_image(image, original_pdf_filename, page_num, input_dir):
    filename = get_dataset_filename(original_pdf_filename, page_num)
    output_path = os.path.join(input_dir, filename)
    cv2.imwrite(output_path, image)


def process_pdf(input_pdf, output_txt, craft_word_model, model, processor, det_model, det_processor, order_model, order_processor, 
                ocr_tokenizer, ocr_model, ocr_image_processor, ocr_image_processor_high,
                chunk_num, chunk_size, is_dataset_mode=False, input_dir=None, original_pdf_filename=None):
    # Ensure necessary folders exist
    ensure_folders_exist()

    # Clear partitions directory
    partitions_dir = os.path.join(os.getcwd(), "partitions")
    for subfolder in os.listdir(partitions_dir):
        subfolder_path = os.path.join(partitions_dir, subfolder)
        if os.path.isdir(subfolder_path):
            shutil.rmtree(subfolder_path)

    # Get the base name and title of the PDF
    pdf_base_name = os.path.basename(input_pdf)
    pdf_title = os.path.splitext(pdf_base_name)[0]

    # Open the input PDF file
    doc = fitz.open(input_pdf)

    # Get the total number of pages in the PDF
    total_pages = doc.page_count

    # Create the 'partitions' folder if it doesn't exist
    partitions_folder = 'partitions'
    os.makedirs(partitions_folder, exist_ok=True)

    # Calculate the starting page number for this chunk
    start_page_num = (chunk_num - 1) * chunk_size + 1

    # Initialize list to store CRAFT bounding boxes for all pages
    all_craft_bboxes = []

    # Process each page of the PDF
    for page_num in tqdm(range(total_pages), desc='Processing Pages'):
        # Create a subfolder for the current page
        page_folder = os.path.join(partitions_folder, f'{page_num:04d}')
        os.makedirs(page_folder, exist_ok=True)

        # Load the page
        page = doc.load_page(page_num)

        # Apply zooming to the page
        zoom_factor = 4
        zoom_mat = fitz.Matrix(zoom_factor, zoom_factor)
        zoomed_pix = page.get_pixmap(matrix=zoom_mat)

        # Convert the zoomed pixmap to a NumPy array with proper color conversion
        zoomed_img = cv2.cvtColor(np.frombuffer(zoomed_pix.samples, np.uint8).reshape(zoomed_pix.height, zoomed_pix.width, zoomed_pix.n), cv2.COLOR_BGR2RGB)

        # Resize the image using OpenCV
        scaled_height = 2048
        scaled_width = int(zoomed_img.shape[1] * (scaled_height / zoomed_img.shape[0]))
        scaled_img = cv2.resize(zoomed_img, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

        # Save the scaled image as "raw.png" in the page folder
        raw_path = os.path.join(page_folder, 'raw.png')
        cv2.imwrite(raw_path, scaled_img)

        # If in dataset mode, save the additional image in the input directory
        if is_dataset_mode:
            dataset_filename = f"{os.path.splitext(os.path.basename(original_pdf_filename))[0]}_{start_page_num + page_num}.png"
            dataset_path = os.path.join(input_dir, dataset_filename)
            cv2.imwrite(dataset_path, scaled_img)

        # Run CRAFT on the scaled image
        try:
            polygons = craft_word_model.get_polygons(scaled_img)
            craft_bboxes = [polygon_to_bbox(polygon) for polygon in polygons]
            all_craft_bboxes.append(craft_bboxes)
            
            # Create denoised image using CRAFT bboxes
            heights = [calculate_bbox_height(bbox) for bbox in craft_bboxes]
            widths = [calculate_bbox_width(bbox) for bbox in craft_bboxes]
            average_height = sum(heights) / len(heights) if len(heights) > 0 else 0
            average_width = sum(widths) / len(widths) if len(widths) > 0 else 0
            
            denoised_image = np.ones((scaled_img.shape[0], scaled_img.shape[1] * 2, 3), dtype=np.uint8) * 255
            denoised_image = draw_bounding_boxes(denoised_image, craft_bboxes, average_height, average_width)
            
            denoised_path = os.path.join(page_folder, 'denoised.png')
            cv2.imwrite(denoised_path, denoised_image)
        except Exception as e:
            print(f"Error running CRAFT on page {page_num + 1}: {str(e)}")
            all_craft_bboxes.append([])  # Add an empty list for this page

    # Close the PDF file
    doc.close()

    # Run the layout parsing function on the page images in the partitions directory
    print("Running layout parsing")
    image_tuples, _, _, _, _ = get_layout(partitions_dir, model, processor, det_model, det_processor, order_model, order_processor, all_craft_bboxes,
                              ocr_tokenizer, ocr_model, ocr_image_processor, ocr_image_processor_high)

    # Process the images using the new OCR function
    html_content = process_images(image_tuples, ocr_tokenizer, ocr_model, ocr_image_processor, ocr_image_processor_high)

    # Write the HTML content to the output file
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("Processing completed.")