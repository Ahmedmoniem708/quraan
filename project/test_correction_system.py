#!/usr/bin/env python3
"""
Test script for the Quran recitation correction system.
This script tests the complete pipeline including API clients, audio processing, and correction functionality.
"""

import os
import sys
import logging
import tempfile
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from quran_muaalem.api_clients import AlQuranAPIClient, TarteelAPIClient
from quran_muaalem.audio_processing import AudioProcessor
from quran_muaalem.correction_pipeline import RecitationCorrector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_api_clients():
    """Test the API clients for AlQuran and Tarteel."""
    logger.info("🧪 Testing API clients...")
    
    try:
        # Test AlQuran API
        logger.info("Testing AlQuran API client...")
        alquran_client = AlQuranAPIClient()
        
        # Test getting ayah data
        ayah_data = alquran_client.get_ayah_data(1, 1, include_audio=True)
        logger.info(f"✅ AlQuran API: Retrieved ayah data for Surah 1, Ayah 1")
        logger.info(f"   Text: {ayah_data.text[:50]}...")
        logger.info(f"   Words count: {len(ayah_data.words)}")
        logger.info(f"   Audio URL: {ayah_data.audio_url is not None}")
        
        # Test Tarteel API
        logger.info("Testing Tarteel API client...")
        tarteel_client = TarteelAPIClient()
        
        # Test getting timing data
        timing_data = tarteel_client.get_ayah_timing_data(1, 1)
        logger.info(f"✅ Tarteel API: Retrieved timing data for Surah 1, Ayah 1")
        logger.info(f"   Timestamps count: {len(timing_data.word_timestamps)}")
        
        if timing_data.word_timestamps:
            first_word = timing_data.word_timestamps[0]
            logger.info(f"   First word timing: word_index={first_word.word_index}, start={first_word.start_time_ms}ms, end={first_word.end_time_ms}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API client test failed: {e}")
        return False


def test_audio_processing():
    """Test audio processing functionality."""
    logger.info("🧪 Testing audio processing...")
    
    try:
        # Create a simple test audio file (sine wave)
        import numpy as np
        import soundfile as sf
        
        # Generate a 3-second sine wave at 440 Hz
        sample_rate = 16000
        duration = 3.0
        frequency = 440.0
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            temp_audio_path = temp_file.name
        
        try:
            # Test audio processor
            processor = AudioProcessor()
            
            # Test loading audio
            loaded_audio, sr = processor.load_audio(temp_audio_path)
            logger.info(f"✅ Audio loading: Loaded {len(loaded_audio)} samples at {sr} Hz")
            
            # Test audio segmentation
            segments = processor.segment_audio(loaded_audio, sr, [(0.5, 1.5), (1.5, 2.5)])
            logger.info(f"✅ Audio segmentation: Created {len(segments)} segments")
            
            for i, segment in enumerate(segments):
                logger.info(f"   Segment {i+1}: {len(segment)} samples")
            
            return True
            
        finally:
            # Clean up
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
        
    except Exception as e:
        logger.error(f"❌ Audio processing test failed: {e}")
        return False


def test_correction_pipeline():
    """Test the complete correction pipeline."""
    logger.info("🧪 Testing correction pipeline...")
    
    try:
        # Create a simple test audio file
        import numpy as np
        import soundfile as sf
        
        # Generate a longer audio file for testing
        sample_rate = 16000
        duration = 5.0
        
        # Create a more complex waveform (multiple frequencies)
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = (
            0.3 * np.sin(2 * np.pi * 220 * t) +  # A3
            0.3 * np.sin(2 * np.pi * 330 * t) +  # E4
            0.2 * np.sin(2 * np.pi * 440 * t)    # A4
        )
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, len(audio_data))
        audio_data += noise
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            temp_audio_path = temp_file.name
        
        try:
            # Test correction pipeline
            logger.info("Initializing RecitationCorrector...")
            corrector = RecitationCorrector()
            logger.info("✅ RecitationCorrector initialized successfully")
            
            # Test with a simple ayah (Al-Fatiha, verse 1)
            logger.info("Testing correction with Surah 1, Ayah 1...")
            start_time = time.time()
            
            result = corrector.correct_recitation(
                audio_path=temp_audio_path,
                surah_number=1,
                ayah_number=1,
                start_word_index=0,
                num_words=3  # Test with first 3 words
            )
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Correction completed in {processing_time:.2f} seconds")
            logger.info(f"   Surah: {result.surah_number}, Ayah: {result.ayah_number}")
            logger.info(f"   Ayah text: {result.ayah_text}")
            logger.info(f"   Total words: {result.total_words}")
            logger.info(f"   Errors found: {len(result.errors_found)}")
            logger.info(f"   Overall accuracy: {result.overall_accuracy:.1%}")
            
            # Log error details if any
            for i, error in enumerate(result.errors_found, 1):
                logger.info(f"   Error {i}: Word '{error.word_text}' at index {error.word_index}")
                for detail in error.errors:
                    logger.info(f"     - {detail.error_type}: {detail.description}")
            
            # Test summary generation
            summary = corrector.get_correction_summary(result)
            logger.info(f"   Summary: {summary}")
            
            return True
            
        finally:
            # Clean up
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            
            # Clean up any generated audio files
            for error in result.errors_found if 'result' in locals() else []:
                if error.user_isolated_word_path and os.path.exists(error.user_isolated_word_path):
                    os.unlink(error.user_isolated_word_path)
        
    except Exception as e:
        logger.error(f"❌ Correction pipeline test failed: {e}")
        logger.exception("Full error details:")
        return False


def test_gradio_integration():
    """Test Gradio integration functions."""
    logger.info("🧪 Testing Gradio integration...")
    
    try:
        # Import the gradio app functions
        from gradio_app import process_audio_with_correction, generate_correction_html
        
        # Create a mock result for testing HTML generation
        from quran_muaalem.correction_pipeline import CorrectionResult, WordError, ErrorDetails
        
        mock_result = CorrectionResult(
            surah_number=1,
            ayah_number=1,
            ayah_text="بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
            total_words=4,
            errors_found=[],
            overall_accuracy=1.0,
            processing_time=1.5
        )
        
        # Test HTML generation
        html_output = generate_correction_html(mock_result)
        logger.info("✅ HTML generation: Generated correction HTML successfully")
        logger.info(f"   HTML length: {len(html_output)} characters")
        
        # Check if HTML contains expected elements
        expected_elements = ["نتائج فحص التلاوة", "معلومات الآية", "ممتاز", "الملخص"]
        for element in expected_elements:
            if element in html_output:
                logger.info(f"   ✅ Found expected element: {element}")
            else:
                logger.warning(f"   ⚠️ Missing expected element: {element}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Gradio integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting Quran Recitation Correction System Tests")
    logger.info("=" * 60)
    
    tests = [
        ("API Clients", test_api_clients),
        ("Audio Processing", test_audio_processing),
        ("Correction Pipeline", test_correction_pipeline),
        ("Gradio Integration", test_gradio_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed! The system is ready to use.")
        return 0
    else:
        logger.warning(f"⚠️ {total-passed} test(s) failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)