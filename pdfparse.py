from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import vision
from pdf2image import convert_from_path
import requests
import os
from tempfile import NamedTemporaryFile

# Configure Google Vision API credentials using environment variables
if "GOOGLE_CREDENTIALS_JSON" in os.environ:
    temp_key_path = "/tmp/google_key.json"
    with open(temp_key_path, "w") as key_file:
        key_file.write(os.environ["GOOGLE_CREDENTIALS_JSON"])
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_key_path
     
# Initialize FastAPI app
app = FastAPI()

# Request body model
class PDFInput(BaseModel):
    pdf_url: str  # URL of the PDF file

# Helper function to download a file
def download_file(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        temp_file = NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(temp_file.name, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file: {e}")

# Helper function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    client = vision.ImageAnnotatorClient()
    full_text = ""

    for i, image in enumerate(images):
        print(f"Processing page {i + 1}...")

        # Save image temporarily
        temp_image_path = f"temp_page_{i + 1}.jpg"
        image.save(temp_image_path, "JPEG")

        # Load image into Google Vision
        with open(temp_image_path, "rb") as image_file:
            content = image_file.read()
            vision_image = vision.Image(content=content)

        # Detect text
        response = client.text_detection(image=vision_image)
        if response.error.message:
            raise HTTPException(status_code=500, detail=f"Vision API Error: {response.error.message}")

        # Append detected text
        page_text = response.full_text_annotation.text
        full_text += f"--- Page {i + 1} ---\n"
        full_text += page_text + "\n"

        # Clean up temporary image
        os.remove(temp_image_path)

    return full_text

# FastAPI endpoint
@app.post("/process-pdf/")
async def process_pdf(input_data: PDFInput):
    try:
        # Step 1: Download the PDF file
        pdf_path = download_file(input_data.pdf_url)

        # Step 2: Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        # Step 3: Clean up the temporary PDF file
        os.remove(pdf_path)

        # Step 4: Return the extracted text
        return {"extracted_text": extracted_text}

    except HTTPException as http_error:
        raise http_error
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
