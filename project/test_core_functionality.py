#!/usr/bin/env python3
"""
Simple test to verify core functionality without audio processing.
This test focuses on the API clients and basic correction pipeline.
"""

import logging
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quran_muaalem.api_clients import AlQuranAPIClient, TarteelAPIClient
from quran_muaalem.correction_pipeline import RecitationCorrector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_clients():
    """Test API clients functionality."""
    logger.info("🧪 Testing API clients...")
    
    try:
        # Test AlQuran API
        alquran_client = AlQuranAPIClient()
        ayah_data = alquran_client.get_ayah_data(1, 1)  # Al-Fatiha, verse 1
        
        logger.info(f"✅ AlQuran API: Retrieved ayah data")
        logger.info(f"   Text: {ayah_data.text[:50]}...")
        logger.info(f"   Words count: {len(ayah_data.words)}")
        
        # Test Tarteel API (now using local segments.json)
        tarteel_client = TarteelAPIClient()
        timing_data = tarteel_client.get_ayah_timing_data(1, 1)
        
        logger.info(f"✅ Tarteel API: Retrieved timing data")
        logger.info(f"   Timestamps count: {len(timing_data.word_timestamps)}")
        if timing_data.word_timestamps:
            first_word = timing_data.word_timestamps[0]
            logger.info(f"   First word timing: word_index={first_word.word_index}, start={first_word.start_time_ms}ms, end={first_word.end_time_ms}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API clients test failed: {e}")
        return False

def test_correction_pipeline_basic():
    """Test basic correction pipeline functionality without audio processing."""
    logger.info("🧪 Testing basic correction pipeline...")
    
    try:
        # Initialize corrector
        corrector = RecitationCorrector()
        
        # Test getting ayah and timing data (without audio processing)
        ayah_data, timing_data = corrector._get_ayah_and_timing_data(1, 1)
        
        logger.info(f"✅ Correction pipeline: Retrieved ayah and timing data")
        logger.info(f"   Ayah text: {ayah_data.text[:50]}...")
        logger.info(f"   Timing data duration: {timing_data.duration_ms}ms")
        logger.info(f"   Word timestamps: {len(timing_data.word_timestamps)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Correction pipeline test failed: {e}")
        return False

def main():
    """Run core functionality tests."""
    logger.info("🚀 Starting Core Functionality Tests")
    logger.info("=" * 60)
    
    tests = [
        ("API Clients", test_api_clients),
        ("Basic Correction Pipeline", test_correction_pipeline_basic),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASSED" if i < passed else "❌ FAILED"
        logger.info(f"{test_name:<25}: {status}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All core functionality tests passed!")
        return 0
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())