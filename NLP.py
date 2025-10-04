import cv2
import numpy as np
from PIL import Image
from skimage.filters import unsharp_mask # Fixed import
import pytesseract
import spacy
import os
import json
import glob
from pdf2image import convert_from_path 


INPUT_FOLDER_PATH = '/Users/midididdisushanth/UipAth Project/10 - Contracts' 

BASE_OUTPUT_DIR = 'Processed Outputs' 
# Separate directories for each output type
IMAGE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, '/Users/midididdisushanth/UipAth Project/PreProcessed outputs')
TEXT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, '/Users/midididdisushanth/UipAth Project/OCR text files')
NLP_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, '/Users/midididdisushanth/UipAth Project/processed outputs')

OUTPUT_JSON_FILE = os.path.join(NLP_OUTPUT_DIR, 'extracted_contract_data.json')


try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully.")
except OSError:
    print("❌ spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    exit()



def preprocess_image_for_ocr(pil_image):
    """Applies preprocessing to a PIL Image object."""
    img = np.array(pil_image.convert('RGB')) 
    img = img[:, :, ::-1].copy() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 3) 
    thresh = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    sharpened = unsharp_mask(deskewed, radius=1.0, amount=1.0, channel_axis=None, preserve_range=False)
    sharpened_8bit = (sharpened * 255).astype(np.uint8)
    return Image.fromarray(sharpened_8bit)

def perform_nlp_analysis(text):
    """Uses spaCy for Named Entity Recognition (NER) on the extracted text."""
    doc = nlp(text)
    contract_entities = {
        "DATE": [], "ORG": [], "PERSON": [], "GPE": [], "MONEY": [],
    }
    for ent in doc.ents:
        if ent.label_ in contract_entities:
            if ent.text.strip() not in contract_entities[ent.label_]:
                contract_entities[ent.label_].append(ent.text.strip())
    return contract_entities



def process_single_contract_page(page_image, base_filename, page_num):
    """Handles the full workflow for a single image/page."""
    
    # 1. Preprocess the PIL Image
    processed_image = preprocess_image_for_ocr(page_image)
    
    # 2. OCR Extraction
    ocr_text = pytesseract.image_to_string(processed_image)
    
    # 3. NLP Analysis
    extracted_data = perform_nlp_analysis(ocr_text)

    # --- Saving Files to New Directories ---
    
    # Save 1: Preprocessed Image
    unique_img_filename = f"{base_filename}_page{page_num:02d}_processed.png"
    output_img_path = os.path.join(IMAGE_OUTPUT_DIR, unique_img_filename)
    processed_image.save(output_img_path)
    
    # Save 2: Raw OCR Text
    unique_text_filename = f"{base_filename}_page{page_num:02d}_raw_ocr.txt"
    output_text_path = os.path.join(TEXT_OUTPUT_DIR, unique_text_filename)
    with open(output_text_path, 'w', encoding='utf-8') as f:
        f.write(ocr_text)

    # --- Prepare Structured Data (for NLP JSON) ---
    extracted_data['Source_File'] = f"{base_filename}.pdf"
    extracted_data['Page_Number'] = page_num
    extracted_data['Processed_Image_File'] = unique_img_filename
    extracted_data['Raw_OCR_Text_File'] = unique_text_filename # Reference to the saved text file
    
    return extracted_data


if __name__ == "__main__":
    
    # --- Setup All Directories ---
    for directory in [IMAGE_OUTPUT_DIR, TEXT_OUTPUT_DIR, NLP_OUTPUT_DIR]:
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Creating output directory: {directory}")

   
    if not os.path.isdir(INPUT_FOLDER_PATH):
       
        os.makedirs(INPUT_FOLDER_PATH, exist_ok=True)
        print(f"🛑 Input directory not found. Created a placeholder: {INPUT_FOLDER_PATH}")
        print("Please place your PDF documents inside this folder and run the script again.")
        exit()
        
    all_pdf_files = glob.glob(os.path.join(INPUT_FOLDER_PATH, '*.pdf'))
    
    if not all_pdf_files:
        print(f"\n🛑 No PDF files found in the folder: '{INPUT_FOLDER_PATH}'. Please check the folder contents.")
    else:
        print(f"\n🔍 Found {len(all_pdf_files)} PDF documents for batch processing.")
        
        all_contract_data = []
        
        for pdf_path in all_pdf_files:
            pdf_filename = os.path.basename(pdf_path)
            base_filename = os.path.splitext(pdf_filename)[0]
            
            print(f"\n--- Processing PDF: {pdf_filename} ---")
            
            try:
                pages = convert_from_path(pdf_path, dpi=300) 
            except Exception as e:
                print(f"❌ Error converting {pdf_filename} to image. Check Poppler installation or file corruption. Error: {e}")
                continue

            for i, page_image in enumerate(pages, 1):
                print(f"  -> Processing Page {i}...")
                result = process_single_contract_page(page_image, base_filename, i)
                if result:
                    all_contract_data.append(result)
            
      
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_contract_data, f, indent=4)
            
        print(f"\n✨ Batch processing finished!")
        print(f"Total pages processed successfully: {len(all_contract_data)}")
        print(f"Outputs are organized into sub-folders within: {BASE_OUTPUT_DIR}")
