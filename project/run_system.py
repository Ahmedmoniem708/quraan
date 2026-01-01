#!/usr/bin/env python3
"""
Quran Recitation Correction System Launcher

This script provides an easy way to launch different components of the system.
"""

import sys
import argparse
import subprocess
import logging
import os
from pathlib import Path

# Fix for LLVM SVML error on Windows - set environment variables early
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
# Additional fixes for LLVM SVML error
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['PYTORCH_DISABLE_CUDA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_gradio_app():
    """Launch the Gradio web interface."""
    logger.info("Starting Gradio web interface...")
    try:
        # Set environment variables for the subprocess as well
        env = os.environ.copy()
        env.update({
            'TF_ENABLE_ONEDNN_OPTS': '0',
            'KMP_DUPLICATE_LIB_OK': 'TRUE',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_THREADING_LAYER': 'GNU',
            'PYTORCH_DISABLE_CUDA': '1',
            'CUDA_VISIBLE_DEVICES': '',
            'MKL_SERVICE_FORCE_INTEL': '1'
        })
        subprocess.run([sys.executable, "-m", "src.quran_muaalem.gradio_app"], check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Gradio app: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("Gradio app stopped by user")
    return True

def run_fastapi_server():
    """Launch the FastAPI backend server."""
    logger.info("Starting FastAPI backend server...")
    try:
        subprocess.run([sys.executable, "-m", "src.quran_muaalem.fastapi_server"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start FastAPI server: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("FastAPI server stopped by user")
    return True

def run_tests():
    """Run the system tests."""
    logger.info("Running system tests...")
    try:
        subprocess.run([sys.executable, "test_correction_system.py"], check=True)
        logger.info("All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {e}")
        return False

def demo_correction():
    """Run a demo correction using sample audio."""
    logger.info("Running demo correction...")
    
    demo_code = '''
from src.quran_muaalem.correction_pipeline import RecitationCorrector
import logging

logging.basicConfig(level=logging.INFO)

# Initialize corrector
print("Initializing Quran recitation corrector...")
corrector = RecitationCorrector()

# Check if test audio exists
import os
test_audio = "assets/test.wav"
if not os.path.exists(test_audio):
    test_audio = "assets/test.mp3"
    if not os.path.exists(test_audio):
        print("No test audio file found. Please add a test audio file to assets/test.wav or assets/test.mp3")
        exit(1)

print(f"Using test audio: {test_audio}")

# Run correction on test audio
try:
    result = corrector.correct_recitation(
        audio_path=test_audio,
        surah_number=1,
        ayah_number=1,
        start_word_index=0,
        num_words=4
    )
    
    print("\\n" + "="*50)
    print("CORRECTION RESULTS")
    print("="*50)
    print(f"Surah: {result.surah_number}, Ayah: {result.ayah_number}")
    print(f"Text: {result.ayah_text}")
    print(f"Overall Accuracy: {result.overall_accuracy:.2%}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    
    if result.errors_found:
        print(f"\\nErrors Found: {len(result.errors_found)}")
        for i, error in enumerate(result.errors_found, 1):
            print(f"\\n{i}. Word {error.word_index}: '{error.word_text}'")
            for detail in error.errors:
                print(f"   - {detail.description}")
                print(f"     Confidence: {detail.confidence:.2%}")
    else:
        print("\\n✅ No errors detected! Excellent recitation!")
        
except Exception as e:
    print(f"Error during correction: {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        subprocess.run([sys.executable, "-c", demo_code], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Demo failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Quran Recitation Correction System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_system.py gradio          # Launch web interface
  python run_system.py fastapi         # Launch API server
  python run_system.py test            # Run tests
  python run_system.py demo            # Run demo correction
        """
    )
    
    parser.add_argument(
        "component",
        choices=["gradio", "fastapi", "test", "demo"],
        help="Component to launch"
    )
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not Path("src/quran_muaalem").exists():
        logger.error("Please run this script from the project root directory")
        sys.exit(1)
    
    success = False
    
    if args.component == "gradio":
        success = run_gradio_app()
    elif args.component == "fastapi":
        success = run_fastapi_server()
    elif args.component == "test":
        success = run_tests()
    elif args.component == "demo":
        success = demo_correction()
    
    if not success:
        sys.exit(1)
    
    logger.info("Operation completed successfully!")

if __name__ == "__main__":
    main()