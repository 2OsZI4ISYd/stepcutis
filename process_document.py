import os
import fitz
from tqdm import tqdm
import cv2
import numpy as np
from projectpackages.CRAFT import CRAFTModel
from layout import get_layout
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

def process_pdf(input_pdf, output_txt, craft_word_model, model, processor, det_model, det_processor, det_model2, det_processor2, order_model, order_processor, chunk_num, chunk_size, is_dataset_mode=False, input_dir=None, original_pdf_filename=None):
    # Ensure necessary folders exist
    ensure_folders_exist()

    # Create the path to the "partitions" folder
    partitions_dir = os.path.join(os.getcwd(), "partitions")

    # Iterate over the subfolders in the "partitions" folder
    for subfolder in os.listdir(partitions_dir):
        subfolder_path = os.path.join(partitions_dir, subfolder)
        
        # Check if the item is a directory (subfolder)
        if os.path.isdir(subfolder_path):
            # Delete the subfolder and its contents
            shutil.rmtree(subfolder_path)

    # Create the path to the "regionimages" folder
    regionimages_dir = os.path.join(os.getcwd(), "regionimages")

    # Iterate over the files in the "regionimages" folder
    for file in os.listdir(regionimages_dir):
        file_path = os.path.join(regionimages_dir, file)
        
        # Check if the item is a file
        if os.path.isfile(file_path):
            # Delete the file
            os.remove(file_path)

    # Get the base name and title of the PDF
    pdf_base_name = os.path.basename(input_pdf)
    pdf_title = os.path.splitext(pdf_base_name)[0]

    # Open the input PDF file
    doc = fitz.open(input_pdf)

    # Get the total number of pages in the PDF
    total_pages = doc.page_count

    # Create the 'partitions' folder if it doesn't exist
    partitions_folder = 'partitions'
    create_folder(partitions_folder)

    # Calculate the starting page number for this chunk
    start_page_num = (chunk_num - 1) * chunk_size + 1

    # Initialize list to store CRAFT bounding boxes for all pages
    all_craft_bboxes = []

    # Process each page of the PDF
    for page_num in tqdm(range(total_pages), desc='Processing Pages'):
        # Create a subfolder for the current page
        page_folder = os.path.join(partitions_folder, f'{page_num:04d}')
        create_folder(page_folder)

        # Load the page
        page = doc.load_page(page_num)

        # Apply zooming to the page
        zoom_factor = 4  # Adjust the zoom factor as needed
        zoom_mat = fitz.Matrix(zoom_factor, zoom_factor)
        zoomed_pix = page.get_pixmap(matrix=zoom_mat)

        # Get the height of the zoomed image
        zoomed_height = zoomed_pix.height

        # Calculate the scale factor to make the height 2048
        scale_factor = 1 #2048 / zoomed_height

        # Convert the zoomed pixmap to a NumPy array with proper color conversion
        zoomed_img = cv2.cvtColor(np.frombuffer(zoomed_pix.samples, np.uint8).reshape(zoomed_pix.height, zoomed_pix.width, zoomed_pix.n), cv2.COLOR_BGR2RGB)

        # Resize the image using OpenCV
        scaled_height = 2048
        scaled_width = int(zoomed_img.shape[1] * (scaled_height / zoomed_img.shape[0]))
        scaled_img = cv2.resize(zoomed_img, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

        # Save the scaled image as "raw.png" in the page folder
        raw_path = os.path.join(page_folder, 'raw.png')
        cv2.imwrite(raw_path, scaled_img)
        
        # Get the absolute path of the "raw.png" image
        absolute_raw_path = os.path.abspath(raw_path)

        # If in dataset mode, save the additional image in the input directory
        if is_dataset_mode:
            save_dataset_image(scaled_img, original_pdf_filename, start_page_num + page_num, input_dir)

        # Run CRAFT on the scaled image
        # print(f"Running CRAFT on page {page_num + 1}")
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
    get_layout(partitions_dir, model, processor, det_model, det_processor, det_model2, det_processor2, order_model, order_processor, all_craft_bboxes)

    # Run the OCR script on the images in the regionimages directory 
    # subprocess.run(['node', 'ocr.js', output_txt, str(chunk_num), str(chunk_size)], universal_newlines=True)
    subprocess.run(['python', './projectpackages/LakeOCR/GOT/demo/inference.py', output_txt, str(chunk_num), str(chunk_size)], universal_newlines=True)

    print("Processing completed.")