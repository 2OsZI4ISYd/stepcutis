from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
from projectpackages.surya.detection import batch_text_detection
from projectpackages.surya.layout import batch_layout_detection
from projectpackages.surya.ordering import batch_ordering
from projectpackages.surya.tables import batch_table_recognition
from projectpackages.surya.schema import TextDetectionResult, PolygonBox, ColumnLine

def extract_bounding_boxes(layout_predictions, is_sparse=False):
    bounding_boxes = []
    for layout_result in layout_predictions:
        for layout_box in layout_result.bboxes:
            if layout_box.label in ['Picture', 'Figure']:
                label = 'Caption'
            else:
                label = layout_box.label
            bounding_boxes.append(layout_box.bbox + [label])
    return bounding_boxes

def filter_text_labels(bounding_boxes):
    return [box for box in bounding_boxes if box[4] != 'Text']

def shrink_bbox_horizontally(bbox):
    x1, y1, x2, y2 = bbox
    shift_value = abs(x1 - x2) / 3
    new_x1 = x1 + shift_value
    new_x2 = x2 - shift_value
    return [new_x1, y1, new_x2, y2]

def get_shortened_label(label):
    label_map = {
        'Caption': 'c', 'Footnote': 'fo', 'Formula': 'fr', 'List-item': 'l',
        'Page-footer': 'pf', 'Page-header': 'ph', 'Picture': 'p', 'Figure': 'f',
        'Section-header': 's', 'Table': 'a', 'Text': 'e', 'Title': 'i'
    }
    return label_map.get(label, 'u')

def assign_boxIDs(bounding_boxes):
    return [bbox + [i] for i, bbox in enumerate(bounding_boxes)]

def valueInRange(value, min_val, max_val):
    return (min_val <= value <= max_val)

def rectOverlap(box1, box2):
    x1_l, y1_b, x1_r, y1_t = box1[:4]
    x2_l, y2_b, x2_r, y2_t = box2[:4]

    xOverlap = valueInRange(x1_l, x2_l, x2_r) or valueInRange(x2_l, x1_l, x1_r)
    yOverlap = valueInRange(y1_b, y2_b, y2_t) or valueInRange(y2_b, y1_b, y1_t)

    return xOverlap and yOverlap

def full_encapsulation(primary_box, secondary_box):
    x11, y11, x12, y12 = primary_box[:4]
    x21, y21, x22, y22 = secondary_box[:4]
    return (x11 <= x21 < x22 <= x12) and (y11 <= y21 < y22 <= y12)

def consolidate_regions(sparse_line_regions, layout_regions):
    to_delete_from_layout = set()
    to_delete_from_sparse = set()

    for i, primary_box in enumerate(sparse_line_regions):
        for j, secondary_box in enumerate(layout_regions):
            if (primary_box[4] in ['Title', 'Section-header', 'Page-header', 'List-item'] and rectOverlap(primary_box[:4], secondary_box[:4]) and 2 * abs(int(primary_box[1]) - int(primary_box[3])) > abs(int(secondary_box[1]) - int(secondary_box[3]))) or (primary_box[4] in ['Table', 'Caption', 'Footnote', 'Page-footer'] and full_encapsulation(primary_box[:4], secondary_box[:4])):
                to_delete_from_layout.add(j)

    consolidated_regions = [box for i, box in enumerate(sparse_line_regions) if i not in to_delete_from_sparse]
    consolidated_regions += [box for i, box in enumerate(layout_regions) if i not in to_delete_from_layout]

    return consolidated_regions

def get_layout(partitions_directory, model, processor, det_model, det_processor, table_model, table_processor, order_model, order_processor, craft_bboxes,
               ocr_tokenizer, ocr_model, ocr_image_processor, ocr_image_processor_high):    
    
    partitions_directory = os.path.join(os.getcwd(), 'partitions')
    page_folders = sorted([folder for folder in os.listdir(partitions_directory) if folder.isdigit()], key=lambda x: int(x))

    denoised_images = []
    normal_images = []
    raw_images = []

    for page_folder in page_folders:
        raw_image_path = os.path.join(partitions_directory, page_folder, 'raw.png')
        denoised_image_path = os.path.join(partitions_directory, page_folder, 'denoised.png')
        
        denoised_image = Image.open(denoised_image_path)
        raw_image = cv2.imread(raw_image_path)
        normal_image = Image.open(raw_image_path)

        normal_width, normal_height = normal_image.size
        denoised_image = denoised_image.resize((normal_width, normal_height), Image.LANCZOS)
        
        denoised_images.append(denoised_image)
        normal_images.append(normal_image)
        raw_images.append(raw_image)

    # Create craft_line_predictions
    craft_line_predictions = []
    for normal_image, page_craft_bboxes in zip(normal_images, craft_bboxes):
        bboxes = []
        for craft_bbox in page_craft_bboxes:
            x1, y1, x2, y2 = craft_bbox
            polygon_box = PolygonBox(
                polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                confidence=0.6
            )
            bboxes.append(polygon_box)
        
        craft_line_prediction = TextDetectionResult(
            bboxes=bboxes,
            vertical_lines=[],
            heatmap=None,
            affinity_map=None,
            image_bbox=[0, 0, normal_image.width, normal_image.height]
        )
        craft_line_predictions.append(craft_line_prediction)

    dense_line_predictions = batch_text_detection(denoised_images, det_model, det_processor)
    sparse_line_predictions = batch_text_detection(normal_images, det_model, det_processor)
    layout_predictions = batch_layout_detection(denoised_images, model, processor, dense_line_predictions)
    sparse_layout_predictions = batch_layout_detection(normal_images, model, processor, sparse_line_predictions)

    original_layout_regions_scaled_list = []
    normal_image_cv_list = []
    vertically_swelled_layout_regions_scaled_list = []
    normal_image_cv_original_list = []
    vertically_swelled_layout_regions_coords_list = []

    for page_number, normal_image, denoised_image, layout_prediction, sparse_layout_prediction, sparse_line_prediction, craft_line_prediction, page_craft_bboxes in tqdm(zip(range(len(page_folders)), normal_images, denoised_images, layout_predictions, sparse_layout_predictions, sparse_line_predictions, craft_line_predictions, craft_bboxes), total=len(page_folders), desc="Processing Pages"):
        layout_regions = extract_bounding_boxes([layout_prediction], is_sparse=False)
        sparse_line_regions = extract_bounding_boxes([sparse_layout_prediction], is_sparse=True)
        
        image_bbox = [0, 0, normal_image.width, normal_image.height]

        filtered_adjusted_sparse_line_regions = filter_text_labels(sparse_line_regions)
        
        consolidated_regions = consolidate_regions(filtered_adjusted_sparse_line_regions, layout_regions)
        consolidated_regions_with_ids = assign_boxIDs(consolidated_regions)
        
        if len(consolidated_regions_with_ids) >= 255:
            x_min = min(bbox[0] for bbox in consolidated_regions_with_ids)
            y_min = min(bbox[1] for bbox in consolidated_regions_with_ids)
            x_max = max(bbox[2] for bbox in consolidated_regions_with_ids)
            y_max = max(bbox[3] for bbox in consolidated_regions_with_ids)
            consolidated_regions_with_ids = [[x_min, y_min, x_max, y_max, 'Text', 0]]

        original_layout_regions_scaled = consolidated_regions_with_ids
        
        normal_image_cv = cv2.cvtColor(np.array(normal_image), cv2.COLOR_RGB2BGR)
    
        vertically_swelled_layout_regions_scaled = original_layout_regions_scaled.copy()
    
        vertically_swelled_layout_regions_coords = [bbox[:4] for bbox in vertically_swelled_layout_regions_scaled]
    
        original_layout_regions_scaled_list.append(original_layout_regions_scaled)
        normal_image_cv_list.append(normal_image_cv)
        normal_image_cv_original_list.append(normal_image_cv.copy())
        vertically_swelled_layout_regions_scaled_list.append(vertically_swelled_layout_regions_scaled)
        vertically_swelled_layout_regions_coords_list.append(vertically_swelled_layout_regions_coords)
    
    order_predictions = batch_ordering(normal_images, vertically_swelled_layout_regions_coords_list, order_model, order_processor)

    image_tuples = []

    for page_number, original_layout_regions_scaled, vertically_swelled_layout_regions_scaled_original, order_prediction, raw_image, normal_image_cv, sparse_line_prediction in tqdm(zip(range(len(page_folders)), original_layout_regions_scaled_list, vertically_swelled_layout_regions_scaled_list, order_predictions, raw_images, normal_image_cv_list, sparse_line_predictions), total=len(page_folders), desc="Configuring Region Images"):
        ordered_original_layout_regions_scaled = []
        
        coord_to_box_map = {tuple(box[:4]): box for box in vertically_swelled_layout_regions_scaled_original}

        for order_box in order_prediction.bboxes:
            bbox_tuple = tuple(order_box.bbox)
            position = order_box.position

            if bbox_tuple in coord_to_box_map:
                original_box = coord_to_box_map[bbox_tuple]
                x1, y1, x2, y2, label, box_id = original_box
                ordered_original_layout_regions_scaled.append([x1, y1, x2, y2, label, box_id, position])

        ordered_original_layout_regions_scaled.sort(key=lambda x: x[6])

        normal_cv_height, normal_cv_width, _ = normal_image_cv.shape
        image_bbox = [0, 0, normal_cv_width, normal_cv_height]

        table_images = []
        table_cells_list = []
        table_positions = []

        for region in ordered_original_layout_regions_scaled:
            if region[4] == 'Table':
                x1, y1, x2, y2, _, _, position = region
                table_image = normal_image_cv[int(y1):int(y2), int(x1):int(x2)]
                table_images.append(Image.fromarray(cv2.cvtColor(table_image, cv2.COLOR_BGR2RGB)))
                
                cells_in_table = [cell for cell in sparse_line_prediction.bboxes if full_encapsulation(region[:4], cell.bbox)]
                table_cells = [{"bbox": [cell.bbox[0]-x1, cell.bbox[1]-y1, cell.bbox[2]-x1, cell.bbox[3]-y1]} for cell in cells_in_table]
                table_cells_list.append(table_cells)
                table_positions.append((position, (x1, y1, x2, y2)))
            else:
                x1, y1, x2, y2, label, _, position = region
                region_image = raw_image[int(y1):int(y2), int(x1):int(x2)]
                region_image_rgb = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(region_image_rgb)
                
                image_tuples.append((pil_image, page_number, position, label, None, None))

        if table_images:
            try:
                table_results = batch_table_recognition(table_images, table_cells_list, table_model, table_processor)
                for table_result, (position, (table_x1, table_y1, _, _)) in zip(table_results, table_positions):
                    for cell in table_result.cells:
                        x1, y1, x2, y2 = cell.bbox
                        x1 += table_x1
                        y1 += table_y1
                        x2 += table_x1
                        y2 += table_y1
                        cell_image = raw_image[int(y1):int(y2), int(x1):int(x2)]
                        cell_image_rgb = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(cell_image_rgb)
                        image_tuples.append((pil_image, page_number, position, 'a', cell.row_id, cell.col_id))
            except Exception as e:
                print(f"Error in table recognition for page {page_number}: {str(e)}")
                # Fallback to original table region if recognition fails
                for region in ordered_original_layout_regions_scaled:
                    if region[4] == 'Table':
                        x1, y1, x2, y2, _, _, position = region
                        region_image = raw_image[int(y1):int(y2), int(x1):int(x2)]
                        region_image_rgb = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(region_image_rgb)
                        image_tuples.append((pil_image, page_number, position, 'Table', None, None))

    return image_tuples, ocr_tokenizer, ocr_model, ocr_image_processor, ocr_image_processor_high