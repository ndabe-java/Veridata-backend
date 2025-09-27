from backend.utils.label_generator import generate_label
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import io
 
# Import the updated analysis and PDF functions
from utils.label_generator import analyze_dataset_expanded, generate_pdf_label
 
app = FastAPI()
 
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# In-memory store for analysis results before PDF download
ANALYSIS_CACHE = {}
 
@app.get("/")
def home():
    return {"message": "Welcome to Veridata API"}
 
@app.post("/upload-dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Uploads a CSV, analyzes it, and caches the results.
    """
    try:
        # 1. Read the file contents
        contents = await file.read()
        # Ensure the file handle is reset before reading as CSV
        df = pd.read_csv(io.BytesIO(contents))
        
        # 2. Run the expanded analysis
        analysis_data = analyze_dataset_expanded(df)
        
        if "error" in analysis_data:
            raise Exception(analysis_data["error"])
 
        # 3. Cache the result using a simple unique ID (filename and size)
        # We use a hash or timestamp in a real app, but filename+size works for an MVP
        cache_key = f"{file.filename}_{len(contents)}"
        ANALYSIS_CACHE[cache_key] = analysis_data
        
        # 4. Return the analysis data and the cache key for download
        # The frontend will use the 'nutrition_label' for display and 'download_key' for PDF
        return {
            "filename": file.filename,
            "nutrition_label": analysis_data,
            "download_key": cache_key
        }
        
    except Exception as e:
        # FastAPI's HTTPException returns a proper HTTP error response
        raise HTTPException(status_code=500, detail=f"Analysis error: {e}")
 
 
@app.get("/download-label/")
async def download_label(key: str, filename: str):
    """
    Generates and returns the PDF of the Data Nutrition Label.
    """
    
    analysis_data = ANALYSIS_CACHE.get(key)
    
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Analysis data not found. Please upload the file first.")
 
    try:
        # Generate the PDF into a buffer
        pdf_buffer = generate_pdf_label(analysis_data, dataset_name=filename)
        
        # Set headers for file download
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}_Nutrition_Label.pdf"',
            'Content-Type': 'application/pdf',
        }
        
        # Return the PDF buffer as a streaming response
        return StreamingResponse(
            pdf_buffer,
            headers=headers,
            media_type="application/pdf"
        )
        
    except Exception as e:
        if key in ANALYSIS_CACHE:
             del ANALYSIS_CACHE[key]
        raise HTTPException(status_code=500, detail=f"PDF generation error: {e}")