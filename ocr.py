from PIL import Image
import pytesseract

# --- Step 1: Configuration (Crucial for Windows or custom installations) ---
# Replace this with the actual path to your Tesseract executable.
# Example Windows path: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux/macOS, if Tesseract is in your PATH, you might not need this line.
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>' 

# --- Step 2: Image Path ---
image_path = '/Users/midididdisushanth/Screenshot 2024-02-06 at 7.59.43 PM.png' # Replace with the name/path of your image file

# --- Step 3: Open the image using Pillow ---
try:
    img = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}. Please check the path.")
    exit()

# --- Step 4: Perform OCR to extract text ---
# image_to_string() performs the OCR operation
extracted_text = pytesseract.image_to_string(img)

# --- Step 5: Print the extracted text ---
print("--- Extracted Text ---")
print(extracted_text)