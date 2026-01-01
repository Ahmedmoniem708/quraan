# Quran Recitation Correction System - Setup Guide

## Overview

This system provides AI-powered Quran recitation correction using the Muaalem model, integrated with external APIs and local JSON files for comprehensive word-level error detection and correction.

## System Architecture

### Core Components

1. **Muaalem Model Integration** (`src/quran_muaalem/inference.py`)
   - Uses `obadx/muaalem-model-v3_2` for audio analysis
   - Detects pronunciation errors and tajweed mistakes
   - Returns surah, ayah, and word indices with error descriptions

2. **API Clients** (`src/quran_muaalem/api_clients.py`)
   - **AlQuran API Client**: Fetches ayah text and reference audio
   - **Tarteel API Client**: Provides word-level timing data
   - Integrated data models for seamless API interaction

3. **Audio Processing** (`src/quran_muaalem/audio_processing.py`)
   - Word-level audio segmentation using timestamps
   - Audio format conversion and preprocessing
   - Clip extraction for both student and reference audio

4. **Correction Pipeline** (`src/quran_muaalem/correction_pipeline.py`)
   - Orchestrates the complete workflow
   - Integrates model analysis with API data
   - Generates comprehensive correction results

5. **FastAPI Backend** (`src/quran_muaalem/fastapi_server.py`)
   - RESTful API endpoints for audio upload and processing
   - Handles file uploads and returns structured results
   - CORS-enabled for web integration

6. **Gradio Interface** (`src/quran_muaalem/gradio_app.py`)
   - User-friendly web interface
   - Audio recording and upload capabilities
   - Real-time correction feedback with audio playback

## Prerequisites

### Python Installation
1. Install Python 3.10 or higher from [python.org](https://python.org)
2. Ensure Python is added to your system PATH
3. Verify installation: `python --version`

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

Or install with optional UI dependencies:
```bash
pip install -e ".[ui]"
```

## Local JSON Files

### 1. Surah Audio Metadata (`surah.json`)
Contains reference audio URLs and durations:
```json
{
  "1": {
    "surah_number": 1,
    "audio_url": "https://download.quranicaudio.com/qdc/khalil_al_husary/muallim/1.mp3",
    "duration": 64
  }
}
```

### 2. Segments File (`segments.json`)
Provides word-level timing data:
```json
{
  "1:1": {
    "segments": [
      [1, 0.0, 660.0],
      [2, 660.0, 1410.0],
      [3, 1410.0, 2740.0],
      [4, 2740.0, 4085.0]
    ],
    "duration_sec": 7,
    "duration_ms": 7740,
    "timestamp_from": 0,
    "timestamp_to": 7840
  }
}
```

## Usage

### 1. FastAPI Server
Start the backend server:
```bash
python -m src.quran_muaalem.fastapi_server
```

Server will be available at: `http://localhost:8000`

#### API Endpoints:
- `POST /correct-recitation`: Upload audio for correction
- `POST /correct-ayah`: Analyze complete ayah
- `GET /word-audio/{surah}/{ayah}/{word}`: Get word audio clips
- `GET /ayah-info/{surah}/{ayah}`: Get ayah information

### 2. Gradio Interface
Launch the web interface:
```bash
python -m src.quran_muaalem.gradio_app
```

Features:
- Audio recording and upload
- Surah and ayah selection
- Real-time correction analysis
- Side-by-side audio comparison (student vs. reference)
- Moshaf settings configuration

### 3. Python API
Use the correction pipeline directly:
```python
from src.quran_muaalem.correction_pipeline import RecitationCorrector

# Initialize corrector
corrector = RecitationCorrector()

# Analyze audio file
result = corrector.correct_recitation(
    audio_path="path/to/audio.wav",
    surah_number=1,
    ayah_number=1,
    start_word_index=0,
    num_words=4
)

# Access results
for error in result.errors_found:
    print(f"Error in word {error.word_index}: {error.word_text}")
    for detail in error.errors:
        print(f"  - {detail.description}")
```

## Workflow

1. **Audio Input**: Student records or uploads Quran recitation
2. **Model Analysis**: Muaalem model processes audio and identifies errors
3. **API Integration**: 
   - AlQuran API fetches correct ayah text and words
   - Tarteel API provides word timing data
4. **Audio Processing**: 
   - Extract student's audio segment for incorrect word
   - Fetch and clip reference audio for the same word
5. **Result Generation**: 
   - Compile error descriptions
   - Provide audio clips for comparison
   - Generate correction feedback

## Output Format

The system returns:
- **Correct Word**: The proper text from the Quran
- **Error Description**: Detailed explanation of the mistake
- **Student Audio**: Clipped audio of the mispronounced word
- **Reference Audio**: Correct pronunciation from reference reciter
- **Overall Accuracy**: Percentage score for the recitation

## Configuration

### Moshaf Settings
Customize recitation rules through the Gradio interface or programmatically:
```python
from quran_transcript import MoshafAttributes

moshaf = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=4,
)
```

## Testing

Run the test suite:
```bash
python test_correction_system.py
```

This validates:
- API client functionality
- Audio processing capabilities
- Correction pipeline integration
- Gradio interface components

## Troubleshooting

### Common Issues:

1. **Python Not Found**: Install Python and add to PATH
2. **Model Loading**: Ensure internet connection for model download
3. **Audio Format**: Use WAV or MP3 files, 16kHz sample rate preferred
4. **API Limits**: Respect rate limits for external APIs

### Performance Tips:

1. Use GPU if available for faster model inference
2. Cache model weights locally after first download
3. Preprocess audio to 16kHz mono for optimal performance
4. Use appropriate word ranges to reduce processing time

## Support

For issues or questions:
1. Check the logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure audio files are in supported formats
4. Test with sample audio files first

## License

This project is licensed under the MIT License. See LICENSE file for details.