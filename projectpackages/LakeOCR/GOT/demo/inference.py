import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy as np

import sys
import torch
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup

# Import necessary modules from GOT
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init, KeywordsStoppingCriteria
from GOT.model import GOTQwenForCausalLM
from GOT.model.plug.blip_process import BlipImageEvalProcessor

# Constants
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def initialize_ocr_model():
    disable_torch_init()
    model_name = os.path.join(os.getcwd(), 'GOT-OCR2_0')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
    model.to(device='cuda', dtype=torch.bfloat16)
    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    return tokenizer, model, image_processor, image_processor_high

def process_single_image(image_path, tokenizer, model, image_processor, image_processor_high):
    image = load_image(image_path)
    
    use_im_start_end = True
    image_token_len = 256
    qs = 'OCR with format: '
    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv_mode = "mpt"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])
    image_tensor = image_processor(image)
    image_tensor_1 = image_processor_high(image)
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams=1,
            no_repeat_ngram_size=20,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
        )
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        recognized_text = outputs.strip()

    return recognized_text

def parse_image_filename(filename, chunk_num, chunk_size):
    parts = filename.split('_')
    page_within_chunk = int(parts[0])
    actual_page_num = (int(chunk_num) - 1) * int(chunk_size) + page_within_chunk + 1
    is_table_cell = 'a' in parts
    
    if is_table_cell:
        return {
            'pageNum': actual_page_num,
            'orderNum': int(parts[1]),
            'label': 'a',
            'tableNum': int(parts[1]),
            'row': int(parts[3]),
            'col': int(parts[4].split('.')[0]),
            'fullPath': os.path.join('regionimages', filename),
            'uniqueTableId': f"{actual_page_num}_{parts[1]}"
        }
    else:
        return {
            'pageNum': actual_page_num,
            'orderNum': int(parts[1]),
            'label': parts[2].split('.')[0],
            'fullPath': os.path.join('regionimages', filename)
        }

def manage_table_structure(cell_info, table_structures):
    unique_table_id = cell_info['uniqueTableId']
    row, col = cell_info['row'], cell_info['col']
    ocr_text = cell_info['ocrText']
    
    if unique_table_id not in table_structures:
        table_structures[unique_table_id] = {'cells': {}, 'maxRow': 0, 'maxCol': 0}
    
    table = table_structures[unique_table_id]
    table['cells'][f"{row}_{col}"] = ocr_text
    table['maxRow'] = max(table['maxRow'], row)
    table['maxCol'] = max(table['maxCol'], col)

def create_html_element(label, content):
    soup = BeautifulSoup('', 'html.parser')
    if label == 'c':
        element = soup.new_tag('figcaption', attrs={'class': 'caption'})
    elif label == 'fo':
        element = soup.new_tag('div', attrs={'class': 'footnote'})
    elif label == 'fr':
        element = soup.new_tag('div', attrs={'class': 'formula'})
    elif label == 'l':
        element = soup.new_tag('li', attrs={'class': 'list-item'})
    elif label == 'pf':
        element = soup.new_tag('footer', attrs={'class': 'page-footer'})
    elif label == 'ph':
        element = soup.new_tag('header', attrs={'class': 'page-header'})
    elif label in ['p', 'f']:
        element = soup.new_tag('figure', attrs={'class': 'picture' if label == 'p' else 'figure'})
        img = soup.new_tag('img', src=content)
        element.append(img)
        return element
    elif label == 's':
        element = soup.new_tag('h2', attrs={'class': 'section-header'})
    elif label == 'a':
        return generate_complete_table_html(content)
    elif label == 'e':
        element = soup.new_tag('p', attrs={'class': 'text'})
    elif label == 'i':
        element = soup.new_tag('h1', attrs={'class': 'title'})
    else:
        element = soup.new_tag('div')
    
    element.string = content
    return element

def generate_complete_table_html(table_id):
    soup = BeautifulSoup('', 'html.parser')
    table = soup.new_tag('table', attrs={'class': 'data-table'})
    
    for r in range(table_id['maxRow'] + 1):
        row = soup.new_tag('tr')
        for c in range(table_id['maxCol'] + 1):
            cell = soup.new_tag('td')
            cell.string = table_id['cells'].get(f"{r}_{c}", '')
            row.append(cell)
        table.append(row)
    
    return table

def create_page_number_element(page_num):
    soup = BeautifulSoup('', 'html.parser')
    element = soup.new_tag('div', attrs={'class': 'page-number'})
    element.string = f"Page {page_num}"
    return element

def generate_html_document(processed_results, table_structures):
    soup = BeautifulSoup('<!DOCTYPE html><html><head><meta charset="UTF-8"><style>.page-number { font-weight: bold; margin-bottom: 10px; }</style></head><body></body></html>', 'html.parser')
    
    page_contents = {}
    
    for result in processed_results:
        page_num = result['pageNum']
        if page_num not in page_contents:
            page_contents[page_num] = []
        
        element = create_html_element(result['label'], result['ocrText'])
        page_contents[page_num].append({'orderNum': result['orderNum'], 'html': element})
    
    for table_id, table_data in table_structures.items():
        page_num, order_num = map(int, table_id.split('_'))
        if page_num not in page_contents:
            page_contents[page_num] = []
        table_html = generate_complete_table_html(table_data)
        page_contents[page_num].append({'orderNum': order_num, 'html': table_html})
    
    sorted_page_numbers = sorted(page_contents.keys())
    
    for page_num in sorted_page_numbers:
        page_container = soup.new_tag('div', attrs={'class': 'page'})
        page_number_element = create_page_number_element(page_num)
        page_container.append(page_number_element)
        
        page_contents[page_num].sort(key=lambda x: x['orderNum'])
        for item in page_contents[page_num]:
            page_container.append(item['html'])
        
        soup.body.append(page_container)
    
    return soup.prettify()

def process_pdf(result_file_path, chunk_num, chunk_size):
    tokenizer, model, image_processor, image_processor_high = initialize_ocr_model()
    
    images_directory = 'regionimages'
    image_files = [f for f in os.listdir(images_directory) if f.endswith('.png')]
    
    parsed_images = [parse_image_filename(f, chunk_num, chunk_size) for f in image_files]
    parsed_images.sort(key=lambda x: (x['pageNum'], x['orderNum']))
    
    processed_results = []
    table_structures = {}
    
    for image in parsed_images:
        try:
            ocr_text = process_single_image(image['fullPath'], tokenizer, model, image_processor, image_processor_high)
            result = {**image, 'ocrText': ocr_text}
            if result['label'] == 'a':
                manage_table_structure(result, table_structures)
            else:
                processed_results.append(result)
            print(f"Processed: {image['fullPath']}")
        except Exception as e:
            print(f"Error processing {image['fullPath']}: {str(e)}")
    
    html_content = generate_html_document(processed_results, table_structures)
    
    with open(result_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Clean up GPU resources
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <result_file_path> <chunk_num> <chunk_size>")
        sys.exit(1)
    
    result_file_path = sys.argv[1]
    chunk_num = int(sys.argv[2])
    chunk_size = int(sys.argv[3])
    
    process_pdf(result_file_path, chunk_num, chunk_size)