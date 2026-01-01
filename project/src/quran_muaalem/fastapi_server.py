"""
FastAPI backend for Quran recitation correction system.
"""

import os
import tempfile
import logging
from typing import Optional, List
from pathlib import Path
# os.environ["GRADIO_DISABLE_COMPRESSION"] = "1"

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from quran_transcript import MoshafAttributes
from .correction_pipeline import (
    RecitationCorrector, 
    CorrectionResult, 
    WordError, 
    ErrorDetails,
    correct_single_ayah,
    correct_word_range
)


# Pydantic models for API requests/responses
class CorrectionRequest(BaseModel):
    """Request model for recitation correction."""
    surah_number: int = Field(..., ge=1, le=114, description="Surah number (1-114)")
    ayah_number: int = Field(..., ge=1, description="Ayah number")
    start_word_index: int = Field(0, ge=0, description="Starting word index (0-based)")
    num_words: Optional[int] = Field(None, ge=1, description="Number of words to analyze")


class ErrorDetailsResponse(BaseModel):
    """Response model for error details."""
    error_type: str
    description: str
    confidence: float
    expected_value: str
    actual_value: str
    phoneme_group: str


class WordErrorResponse(BaseModel):
    """Response model for word errors."""
    surah_number: int
    ayah_number: int
    word_index: int
    word_text: str
    errors: List[ErrorDetailsResponse]
    word_audio_available: bool = False
    reference_audio_url: Optional[str] = None


class CorrectionResponse(BaseModel):
    """Response model for correction results."""
    surah_number: int
    ayah_number: int
    ayah_text: str
    total_words: int
    errors_found: List[WordErrorResponse]
    overall_accuracy: float
    processing_time: float
    summary: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    model_loaded: bool


# Initialize FastAPI app
app = FastAPI(
    title="Quran Recitation Correction API",
    description="API for correcting Quran recitation using AI-powered analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global corrector instance
corrector: Optional[RecitationCorrector] = None
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize the correction system on startup."""
    global corrector
    try:
        logger.info("Initializing Quran recitation corrector...")
        corrector = RecitationCorrector()
        logger.info("Corrector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize corrector: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if corrector is not None else "unhealthy",
        message="Quran recitation correction API is running",
        model_loaded=corrector is not None
    )


@app.post("/correct-recitation", response_model=CorrectionResponse)
async def correct_recitation(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file of the recitation"),
    surah_number: int = Form(..., ge=1, le=114, description="Surah number (1-114)"),
    ayah_number: int = Form(..., ge=1, description="Ayah number"),
    start_word_index: int = Form(0, ge=0, description="Starting word index (0-based)"),
    num_words: Optional[int] = Form(None, ge=1, description="Number of words to analyze")
):
    """
    Correct Quran recitation from uploaded audio file.
    
    This endpoint accepts an audio file and analyzes it for recitation errors
    compared to the correct Quranic text and pronunciation rules.
    """
    if corrector is None:
        raise HTTPException(status_code=503, detail="Correction system not initialized")
    
    # Validate audio file
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid audio file format")
    
    temp_audio_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_audio_path = temp_file.name
        
        logger.info(f"Processing recitation for Surah {surah_number}, Ayah {ayah_number}")
        
        # Run correction pipeline
        result = corrector.correct_recitation(
            audio_path=temp_audio_path,
            surah_number=surah_number,
            ayah_number=ayah_number,
            start_word_index=start_word_index,
            num_words=num_words
        )
        
        # Convert result to response format
        response = _convert_correction_result(result)
        
        # Schedule cleanup of temporary files
        background_tasks.add_task(_cleanup_temp_files, temp_audio_path, result.errors_found)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing recitation: {e}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.get("/word-audio/{surah_number}/{ayah_number}/{word_index}")
async def get_word_audio(
    surah_number: int,
    ayah_number: int,
    word_index: int
):
    """
    Get audio segment for a specific word.
    
    This endpoint is used to retrieve the extracted audio segment
    for a word that had errors in the recitation.
    """
    # This would typically retrieve from a cache or database
    # For now, return a placeholder response
    raise HTTPException(status_code=404, detail="Word audio not found")


@app.post("/correct-ayah", response_model=CorrectionResponse)
async def correct_complete_ayah(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(..., description="Audio file of the complete ayah recitation"),
    surah_number: int = Form(..., ge=1, le=114, description="Surah number (1-114)"),
    ayah_number: int = Form(..., ge=1, description="Ayah number")
):
    """
    Correct a complete ayah recitation.
    
    This is a convenience endpoint for analyzing an entire ayah
    without specifying word ranges.
    """
    return await correct_recitation(
        background_tasks=background_tasks,
        audio_file=audio_file,
        surah_number=surah_number,
        ayah_number=ayah_number,
        start_word_index=0,
        num_words=None
    )


@app.get("/ayah-info/{surah_number}/{ayah_number}")
async def get_ayah_info(surah_number: int, ayah_number: int):
    """
    Get information about a specific ayah.
    
    Returns the ayah text and word count for reference.
    """
    if corrector is None:
        raise HTTPException(status_code=503, detail="Correction system not initialized")
    
    try:
        # Get ayah data
        ayah_data = corrector.alquran_client.get_ayah_data(surah_number, ayah_number, include_audio=False)
        
        return {
            "surah_number": surah_number,
            "ayah_number": ayah_number,
            "text": ayah_data.text,
            "words": ayah_data.words,
            "word_count": len(ayah_data.words)
        }
        
    except Exception as e:
        logger.error(f"Error getting ayah info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get ayah info: {str(e)}")


@app.get("/supported-surahs")
async def get_supported_surahs():
    """Get list of supported surahs."""
    # For now, return all surahs (1-114)
    return {
        "surahs": list(range(1, 115)),
        "total_count": 114
    }


def _convert_correction_result(result: CorrectionResult) -> CorrectionResponse:
    """Convert CorrectionResult to API response format."""
    word_errors = []
    
    for word_error in result.errors_found:
        error_details = [
            ErrorDetailsResponse(
                error_type=error.error_type,
                description=error.description,
                confidence=error.confidence,
                expected_value=error.expected_value,
                actual_value=error.actual_value,
                phoneme_group=error.phoneme_group
            )
            for error in word_error.errors
        ]
        
        word_errors.append(WordErrorResponse(
            surah_number=word_error.surah_number,
            ayah_number=word_error.ayah_number,
            word_index=word_error.word_index,
            word_text=word_error.word_text,
            errors=error_details,
            word_audio_available=word_error.user_isolated_word_path is not None,
            reference_audio_url=word_error.reference_audio_url
        ))
    
    # Generate summary
    if corrector:
        summary = corrector.get_correction_summary(result)
    else:
        summary = "Processing completed"
    
    return CorrectionResponse(
        surah_number=result.surah_number,
        ayah_number=result.ayah_number,
        ayah_text=result.ayah_text,
        total_words=result.total_words,
        errors_found=word_errors,
        overall_accuracy=result.overall_accuracy,
        processing_time=result.processing_time,
        summary=summary
    )


def _cleanup_temp_files(temp_audio_path: str, word_errors: List[WordError]):
    """Clean up temporary files created during processing."""
    try:
        # Remove main audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
            logger.debug(f"Cleaned up temp audio: {temp_audio_path}")
        
        # Remove word audio segments
        for word_error in word_errors:
            if word_error.user_isolated_word_path and os.path.exists(word_error.user_isolated_word_path):
                os.unlink(word_error.user_isolated_word_path)
                logger.debug(f"Cleaned up word audio: {word_error.user_isolated_word_path}")
                
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


# Development server function
def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):

    
    """Run the FastAPI server."""
    uvicorn.run(
        "src.quran_muaalem.fastapi_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run server
    run_server(reload=True)