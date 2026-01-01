# Quran Recitation Correction System 🕌

A comprehensive AI-powered system for correcting Quran recitation using the Muaalem model and external APIs. This system provides word-level error detection, audio segmentation, and targeted feedback for Quranic recitation learning.

## 🌟 Features

- **Word-Level Error Detection**: Identifies specific words where recitation errors occur
- **Audio Segmentation**: Extracts audio segments for incorrect words
- **Multi-API Integration**: Uses AlQuran and Tarteel APIs for reference data
- **Dual Interface**: Both Gradio UI and FastAPI backend
- **Comprehensive Analysis**: Provides detailed error descriptions and corrections
- **Reference Audio**: Includes correct pronunciation examples

## 🏗️ System Architecture

### Core Components

1. **Muaalem Model Integration** (`inference.py`)
   - AI model for Quranic recitation analysis
   - Phoneme and Sifat (characteristics) detection

2. **API Clients** (`api_clients.py`)
   - **AlQuranClient**: Retrieves ayah text and audio
   - **TarteelClient**: Provides word-level timestamps

3. **Audio Processing** (`audio_processing.py`)
   - Audio loading, segmentation, and conversion
   - Word-level audio extraction based on timestamps

4. **Correction Pipeline** (`correction_pipeline.py`)
   - Main orchestration of the correction process
   - Error analysis and result generation

5. **FastAPI Backend** (`fastapi_server.py`)
   - RESTful API endpoints for audio processing
   - Scalable backend for integration

6. **Gradio Interface** (`gradio_app.py`)
   - User-friendly web interface
   - Real-time audio recording and analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch (for the Muaalem model)
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd quran-muaalem
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the project**:
   ```bash
   pip install -e .
   ```

### Usage

#### Option 1: Gradio Web Interface

```bash
python gradio_app.py
```

Then open your browser to the displayed URL (typically `http://localhost:7860`).

#### Option 2: FastAPI Backend

```bash
python -m quran_muaalem.fastapi_server
```

API will be available at `http://localhost:8000` with interactive docs at `/docs`.

#### Option 3: Python API

```python
from quran_muaalem.correction_pipeline import RecitationCorrector

# Initialize the corrector
corrector = RecitationCorrector()

# Analyze audio
result = corrector.correct_recitation(
    audio_path="path/to/audio.wav",
    surah_number=1,
    ayah_number=1,
    start_word_index=0,
    num_words=4
)

# Print results
print(f"Accuracy: {result.overall_accuracy:.1%}")
for error in result.errors_found:
    print(f"Error in word '{error.word_text}': {error.errors[0].description}")
```

## 📋 API Reference

### FastAPI Endpoints

#### POST `/correct-recitation`
Analyze uploaded audio for recitation errors.

**Parameters:**
- `audio_file`: Audio file (WAV, MP3, etc.)
- `surah_number`: Surah number (1-114)
- `ayah_number`: Ayah number
- `start_word_index`: Starting word index (0-based)
- `num_words`: Number of words to analyze (optional)

**Response:**
```json
{
  "surah_number": 1,
  "ayah_number": 1,
  "ayah_text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
  "total_words": 4,
  "errors_found": [...],
  "overall_accuracy": 0.95,
  "processing_time": 2.3,
  "summary": "Good recitation with minor pronunciation issues"
}
```

#### GET `/ayah-info/{surah_number}/{ayah_number}`
Get information about a specific ayah.

#### GET `/health`
Health check endpoint.

### Python API

#### `RecitationCorrector`

Main class for recitation correction.

**Methods:**
- `correct_recitation(audio_path, surah_number, ayah_number, start_word_index=0, num_words=None)`
- `get_correction_summary(result)`

#### `AlQuranClient`

Client for AlQuran API.

**Methods:**
- `get_ayah_data(surah_number, ayah_number, include_audio=True)`
- `get_combined_data(surah_number, ayah_number)`

#### `TarteelClient`

Client for Tarteel API.

**Methods:**
- `get_word_timestamps(surah_number, ayah_number)`

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_correction_system.py
```

This tests:
- API client functionality
- Audio processing capabilities
- Complete correction pipeline
- Gradio integration

## 📁 Project Structure

```
quran-muaalem/
├── src/quran_muaalem/
│   ├── inference.py              # Muaalem model integration
│   ├── muaalem_typing.py         # Type definitions
│   ├── api_clients.py            # External API clients
│   ├── audio_processing.py       # Audio utilities
│   ├── correction_pipeline.py    # Main correction logic
│   ├── fastapi_server.py         # FastAPI backend
│   ├── explain.py                # Result explanation
│   └── explain_gradio.py         # Gradio-specific explanations
├── gradio_app.py                 # Gradio web interface
├── test_correction_system.py     # Test suite
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Moshaf Settings

The system supports various Moshaf (Quran manuscript) settings for different recitation styles:

- **Rewaya**: Recitation style (default: "hafs")
- **Madd lengths**: Various elongation rules
- **Ghunnah**: Nasal sound rules
- **Sakt**: Pause rules

Configure these in the Gradio interface or programmatically:

```python
from quran_transcript import MoshafAttributes

moshaf = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    # ... other settings
)
```

## 🌐 External APIs

### AlQuran API
- **Base URL**: `https://api.alquran.cloud/v1/`
- **Purpose**: Ayah text and reference audio
- **Rate Limits**: Reasonable usage expected

### Tarteel API
- **Base URL**: `https://qul.tarteel.ai/`
- **Purpose**: Word-level timestamps
- **Rate Limits**: Reasonable usage expected

## 🎯 Use Cases

1. **Individual Learning**: Students can practice recitation and get immediate feedback
2. **Educational Institutions**: Teachers can use it for systematic recitation assessment
3. **Mobile Apps**: Integration into Quran learning applications
4. **Research**: Analysis of recitation patterns and common errors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Muaalem Model**: obadx/muaalem-model-v3_2
- **AlQuran API**: For providing Quranic text and audio
- **Tarteel**: For word-level timing data
- **Quran Transcript**: For phonetic processing

## 📞 Support

For issues and questions:
1. Check the test suite results
2. Review the logs for error details
3. Open an issue with detailed information

---

**Note**: This system is designed to assist in learning Quranic recitation. It should be used alongside qualified teachers and traditional learning methods.
