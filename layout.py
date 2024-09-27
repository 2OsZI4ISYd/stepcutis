from PIL import Image
import cv2
import numpy as np
import os
from tqdm import tqdm
from projectpackages.surya.detection import batch_text_detection
from projectpackages.surya.layout import batch_layout_detection
from projectpackages.surya.ordering import batch_ordering
from projectpackages.surya.schema import TextDetectionResult, PolygonBox, ColumnLine
from sklearn.cluster import DBSCAN

def extract_bounding_boxes(layout_predictions, is_sparse=False):
    bounding_boxes = []
    for layout_result in layout_predictions:
        for layout_box in layout_result.bboxes:
            if layout_box.label in ['Table', 'Picture', 'Figure']:
                label = 'Caption'
            else:
                label = layout_box.label if is_sparse else 'Text'
            bounding_boxes.append(layout_box.bbox + [label])
    return bounding_boxes

def filter_text_labels(bounding_boxes):
    return [box for box in bounding_boxes if box[4] != 'Text']

def extract_table_cells(table_region, sparse_line_prediction):
    cell_bboxes = [bbox for bbox in sparse_line_prediction.bboxes if full_encapsulation(table_region, bbox.bbox)]
    processed_cells = [cell.bbox for cell in cell_bboxes]
    
    if not processed_cells:
        return [], []

    left_edges = [box[0] for box in processed_cells]
    right_edges = [box[2] for box in processed_cells]
    all_edges = left_edges + right_edges

    clustering = DBSCAN(eps=5, min_samples=1).fit(np.array(all_edges).reshape(-1, 1))
    unique_clusters = np.unique(clustering.labels_)

    column_separators = []
    for cluster in unique_clusters:
        cluster_points = np.array(all_edges)[clustering.labels_ == cluster]
        separator = np.mean(cluster_points)
        column_separators.append(separator)

    column_separators.sort()

    return processed_cells, column_separators

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

def process_table_cells(cell_bboxes, column_separators, page_number, image_number):
    if not cell_bboxes:
        return []

    y_coords = np.array([bbox[1] for bbox in cell_bboxes])
    y_clusters = DBSCAN(eps=5, min_samples=1).fit(y_coords.reshape(-1, 1))
    y_labels = np.unique(y_clusters.labels_)

    processed_cells = []
    for i, (x1, y1, x2, y2) in enumerate(cell_bboxes):
        row = np.where(y_labels == y_clusters.labels_[i])[0][0]
        
        center_x = (x1 + x2) / 2
        column = next((i for i, sep in enumerate(column_separators) if center_x < sep), len(column_separators) - 1)
        
        filename = f"{page_number}_{image_number}_a_{row}_{column}.png"
        processed_cells.append((x1, y1, x2, y2, 'a', filename, row, column))
    
    return processed_cells

def get_layout(partitions_directory, model, processor, det_model, det_processor, order_model, order_processor, craft_bboxes,
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

    original_layout_regions_scaled_with_position_list = []
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

        final_regions = []
        for region in ordered_original_layout_regions_scaled:
            if region[4] == 'Table':
                table_cells, column_separators = extract_table_cells(region[:4], sparse_line_prediction)
                processed_cells = process_table_cells(table_cells, column_separators, page_number, region[6])
                final_regions.extend(processed_cells)
            else:
                final_regions.append(region)

        original_layout_regions_scaled_with_position_list.append(final_regions)

        raw_height, raw_width, _ = raw_image.shape

        x_scale = raw_width / normal_cv_width
        y_scale = raw_height / normal_cv_height

        num_images = len(final_regions)
        num_digits = len(str(num_images))

        for region in final_regions:
            x1, y1, x2, y2, label, *rest = region
            scaled_x1 = int(x1 * x_scale)
            scaled_y1 = int(y1 * y_scale)
            scaled_x2 = int(x2 * x_scale)
            scaled_y2 = int(y2 * y_scale)
            
            if scaled_x2 <= scaled_x1 or scaled_y2 <= scaled_y1:
                print(f"Warning: Skipping invalid region in page {page_number}: {region}")
                continue

            region_image = raw_image[scaled_y1:scaled_y2, scaled_x1:scaled_x2]
            region_image_rgb = cv2.cvtColor(region_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(region_image_rgb)
            
            width, height = pil_image.size
            longest_side, shortest_side = max(width, height), min(width, height)
            
            if longest_side > 0:
                rescale_factor = 1 if longest_side < 1000 else 1000 / longest_side
                needs_rescaling = longest_side >= 1000
                longest_side = min(longest_side, 1000)
                
                new_width = max(1, int(width * rescale_factor))
                new_height = max(1, int(height * rescale_factor))
                
                try:
                    wide_length, narrow_length = int(longest_side * 1.5), int(longest_side + rescale_factor * shortest_side * 0.5)
                    canvas = Image.new('RGB', (normal_image.width, normal_image.height), color='white')
                    paste_x = (normal_image.width - pil_image.width) // 2
                    paste_y = (normal_image.height - pil_image.height) // 2
                    canvas.paste(pil_image, (paste_x, paste_y))
                    pil_image = canvas
                    
                except Exception as e:
                    print(f"Error processing image for region in page {page_number}: {str(e)}. Skipping this region.")
                    continue
            else:
                print(f"Warning: Skipping region with invalid dimensions in page {page_number}: {region}")
                continue
            
            if label == 'a':  # Table cell
                row, col = rest[2], rest[3]
                image_tuples.append((pil_image, page_number, rest[1], label, row, col))
            else:
                position = rest[1] if len(rest) > 1 else 0
                image_tuples.append((pil_image, page_number, position, label, None, None))

    return image_tuples, ocr_tokenizer, ocr_model, ocr_image_processor, ocr_image_processor_high
