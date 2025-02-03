from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from dotenv import load_dotenv
import base64
import io
from concurrent.futures import ThreadPoolExecutor
import logging
from pdf2image import convert_from_bytes
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('invoice_processor.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate environment variables
required_env_vars = ["AZURE_ENDPOINT", "AZURE_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

app = FastAPI()

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Configuration
try:
    document_analysis_client = DocumentAnalysisClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
    )
    logger.info("Successfully initialized Document Analysis Client")
except Exception as e:
    logger.error(f"Failed to initialize Document Analysis Client: {str(e)}")
    raise

async def extract_first_page(file_content: bytes) -> str:
    """Extract first page from PDF and return as a high quality thumbnail image in base64 encoded string."""
    def _extract():
        try:
            logger.info(f"Starting PDF first page extraction with content size: {len(file_content)} bytes")
            
            # Increase DPI for higher initial quality
            images = convert_from_bytes(
                file_content,
                first_page=1,
                last_page=1,
                dpi=800,  # Increased from 350 to 600 for better quality
                fmt='PNG'
            )
            
            if not images:
                logger.error("No images were converted from PDF")
                return None
            
            logger.info(f"Successfully converted PDF to {len(images)} images")
            
            # Get the first image
            first_page = images[0]
            logger.info(f"First page original size: {first_page.size}")
            
            # Increase max size for better quality thumbnail
            max_size = (800, 800)  # Increased from 800x800 to 1200x1200
            
            # Use high-quality resampling
            first_page.thumbnail(max_size, Image.Resampling.LANCZOS)
            logger.info(f"Thumbnail size after resize: {first_page.size}")
            
            # Save with optimal quality settings
            img_buffer = io.BytesIO()
            first_page.save(
                img_buffer,
                format='PNG',
                optimize=True,
                quality=100,
                subsampling=0,  # Disable chroma subsampling
                icc_profile=first_page.info.get('icc_profile', ''),  # Preserve color profile
                dpi=(800, 800)  # Maintain high DPI in saved image
            )
            img_buffer.seek(0)
            
            # Convert to base64
            thumbnail = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            logger.info(f"Generated thumbnail of size: {len(thumbnail)} bytes")
            
            return thumbnail
                
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}", exc_info=True)
            return None

    return await asyncio.get_event_loop().run_in_executor(executor, _extract)

async def process_invoice(file_content: bytes, filename: str) -> dict:
    """Process invoice with Azure Document Intelligence."""
    try:
        logger.info(f"Starting invoice analysis for file: {filename}")
        
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-invoice",
            document=file_content
        )
        result = poller.result()
        
        # Field mapping for the new format
        field_mapping = {
            'CustomerName': 'Customer Name',
            'CustomerAddress': 'Customer Address',
            'InvoiceDate': 'Invoice Date',
            'InvoiceId': 'Invoice Id',
            'InvoiceTotal': 'Invoice Total'
        }
        
        # Initialize response structure
        invoice_data = {}
        
        # Process document if available
        if hasattr(result, 'documents') and result.documents:
            document = result.documents[0]
            fields = document.fields

            # Process each field with the new structure
            for api_field, display_field in field_mapping.items():
                if api_field in fields:
                    invoice_data[display_field] = {
                        "value": fields[api_field].content,
                        "confidence": fields[api_field].confidence
                    }
                else:
                    invoice_data[display_field] = {
                        "value": None,
                        "confidence": 0.0
                    }

        logger.info(f"Extracted invoice data: {invoice_data}")
        return invoice_data

    except Exception as e:
        logger.error(f"Invoice processing error for {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-document")
async def process_document(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    """Process a single invoice document with thumbnail generation."""
    try:
        logger.info(f"Processing invoice: {file.filename}")
        
        file_content = await file.read()
        metadata_dict = json.loads(metadata)
        
        # Process invoice and generate thumbnail concurrently
        extracted_data_task = process_invoice(file_content, file.filename)
        thumbnail_task = extract_first_page(file_content)
        
        # Wait for both tasks to complete
        extracted_data, thumbnail = await asyncio.gather(
            extracted_data_task,
            thumbnail_task
        )
        
        if thumbnail is None:
            logger.warning(f"Thumbnail generation failed for {file.filename}")
        
        response_data = {
            "filename": file.filename,
            "extracted_data": extracted_data,
            "thumbnail": thumbnail,
            "metadata": metadata_dict
        }
        
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    