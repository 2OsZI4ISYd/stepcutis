import os
import sys
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup
from projectpackages.LakeOCR.GOT.utils.conversation import conv_templates, SeparatorStyle
from projectpackages.LakeOCR.GOT.utils.utils import disable_torch_init, KeywordsStoppingCriteria
from projectpackages.LakeOCR.GOT.model import GOTQwenForCausalLM
from projectpackages.LakeOCR.GOT.model.plug.blip_process import BlipImageEvalProcessor
from tqdm import tqdm

# Constants
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'
DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'

def initialize_ocr_model():
    disable_torch_init()
    model_name = os.path.join(os.getcwd(), 'GOT-OCR2_0')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()
    model.to(device='cuda', dtype=torch.float16)
    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    return tokenizer, model, image_processor, image_processor_high

def is_rich_image(label):
    return label == 'fr'  # 'fr' stands for formula

def process_single_image(pil_image, tokenizer, model, image_processor, image_processor_high, image_label):
    use_im_start_end = True
    image_token_len = 256
    
    is_rich = is_rich_image(image_label)
    qs = 'OCR with format: ' if is_rich else 'OCR: '
    
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
    image_tensor = image_processor(pil_image)
    image_tensor_1 = image_processor_high(pil_image)
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.autocast("cuda", dtype=torch.float16):
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

def process_images(image_tuples):
    tokenizer, model, image_processor, image_processor_high = initialize_ocr_model()
    
    processed_results = []
    table_structures = {}
    
    for pil_image, page_num, order_num, label, row, col in tqdm(image_tuples, desc="Processing images", unit="image"):
        try:
            ocr_text = process_single_image(pil_image, tokenizer, model, image_processor, image_processor_high, label)
            result = {
                'pageNum': page_num,
                'orderNum': order_num,
                'label': label,
                'ocrText': ocr_text
            }
            if label == 'a':
                unique_table_id = f"{page_num}_{order_num}"
                manage_table_structure({**result, 'row': row, 'col': col, 'uniqueTableId': unique_table_id}, table_structures)
            else:
                processed_results.append(result)
        except Exception as e:
            tqdm.write(f"Error processing image on page {page_num}, order {order_num}: {str(e)}")
    
    html_content = generate_html_document(processed_results, table_structures)

    # Clean up GPU resources
    del model
    torch.cuda.empty_cache()

    return html_content