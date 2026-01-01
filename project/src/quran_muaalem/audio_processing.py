"""
Audio processing utilities for Quran recitation correction system.
"""

import os
import sys

# Fix for LLVM SVML error on Windows
# Set environment variables before importing librosa
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Additional fixes for LLVM SVML error
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['PYTORCH_DISABLE_CUDA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Try to disable Intel MKL optimizations that cause LLVM errors
try:
    import mkl
    mkl.set_num_threads(1)
except ImportError:
    pass

import numpy as np
import tempfile
import logging
from typing import Tuple, Union, Optional
from pathlib import Path
from .api_clients import WordTimestamp

# Import librosa with error handling
try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except Exception as e:
    logging.warning(f"Audio processing libraries not available: {e}")
    AUDIO_PROCESSING_AVAILABLE = False


class AudioProcessor:
    """Handles audio processing operations including segmentation and format conversion."""
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize AudioProcessor.
        
        Args:
            target_sample_rate: Target sample rate for audio processing (default: 16000)
        """
        self.target_sample_rate = target_sample_rate
        self.logger = logging.getLogger(__name__)
        
        if not AUDIO_PROCESSING_AVAILABLE:
            self.logger.warning("Audio processing libraries not available. Audio features will be limited.")
    
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to target sample rate.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            raise RuntimeError("Audio processing libraries not available. Cannot load audio.")
            
        try:
            audio_data, original_sr = librosa.load(
                audio_path, 
                sr=self.target_sample_rate, 
                mono=True
            )
            
            self.logger.debug(f"Loaded audio: {len(audio_data)} samples at {self.target_sample_rate}Hz")
            return audio_data, self.target_sample_rate
            
        except Exception as e:
            self.logger.error(f"Error loading audio file {audio_path}: {e}")
            raise Exception(f"Failed to load audio: {e}")
    
    def extract_word_segment(
        self, 
        audio_data: np.ndarray, 
        sample_rate: int,
        word_timestamp: WordTimestamp,
        padding_ms: int = 100
    ) -> np.ndarray:
        """
        Extract a word segment from audio data based on timestamp.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            word_timestamp: WordTimestamp object containing start/end times
            padding_ms: Additional padding in milliseconds (default: 100ms)
            
        Returns:
            Extracted audio segment as numpy array
        """
        try:
            # Convert milliseconds to samples
            start_sample = int((word_timestamp.start_time_ms - padding_ms) * sample_rate / 1000)
            end_sample = int((word_timestamp.end_time_ms + padding_ms) * sample_rate / 1000)
            
            # Ensure bounds are within audio length
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            
            if start_sample >= end_sample:
                raise ValueError(f"Invalid timestamp range: start={start_sample}, end={end_sample}")
            
            # Extract segment
            segment = audio_data[start_sample:end_sample]
            
            self.logger.debug(f"Extracted segment: {len(segment)} samples ({len(segment)/sample_rate:.2f}s)")
            return segment
            
        except Exception as e:
            self.logger.error(f"Error extracting word segment: {e}")
            raise Exception(f"Failed to extract word segment: {e}")
    
    def save_audio_segment(
        self, 
        audio_segment: np.ndarray, 
        sample_rate: int,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Save audio segment to file.
        
        Args:
            audio_segment: Audio data as numpy array
            sample_rate: Sample rate of the audio
            output_path: Output file path (if None, creates temporary file)
            
        Returns:
            Path to saved audio file
        """
        try:
            if output_path is None:
                # Create temporary file
                temp_fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='word_segment_')
                os.close(temp_fd)  # Close file descriptor, we'll use the path
            
            # Save audio using soundfile
            sf.write(output_path, audio_segment, sample_rate)
            
            self.logger.debug(f"Saved audio segment to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error saving audio segment: {e}")
            raise Exception(f"Failed to save audio segment: {e}")
    
    def extract_and_save_word(
        self,
        audio_path: Union[str, Path],
        word_timestamp: WordTimestamp,
        output_path: Optional[Union[str, Path]] = None,
        padding_ms: int = 100
    ) -> str:
        """
        Complete pipeline to extract and save a word segment from audio file.
        
        Args:
            audio_path: Path to input audio file
            word_timestamp: WordTimestamp object
            output_path: Output file path (if None, creates temporary file)
            padding_ms: Additional padding in milliseconds
            
        Returns:
            Path to saved word segment audio file
        """
        try:
            # Load audio
            audio_data, sample_rate = self.load_audio(audio_path)
            
            # Extract word segment
            word_segment = self.extract_word_segment(
                audio_data, sample_rate, word_timestamp, padding_ms
            )
            
            # Save segment
            output_file = self.save_audio_segment(word_segment, sample_rate, output_path)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error in extract_and_save_word pipeline: {e}")
            raise
    
    def get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            self.logger.error(f"Error getting audio duration: {e}")
            raise Exception(f"Failed to get audio duration: {e}")
    
    def convert_audio_format(
        self, 
        input_path: Union[str, Path], 
        output_path: Union[str, Path],
        target_format: str = 'wav'
    ) -> str:
        """
        Convert audio file to different format.
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_format: Target format (wav, mp3, etc.)
            
        Returns:
            Path to converted audio file
        """
        try:
            # Load audio
            audio_data, sample_rate = self.load_audio(input_path)
            
            # Save in target format
            if target_format.lower() == 'wav':
                sf.write(output_path, audio_data, sample_rate)
            else:
                # For other formats, you might need additional libraries like pydub
                raise NotImplementedError(f"Format {target_format} not yet supported")
            
            self.logger.debug(f"Converted audio to {target_format}: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error converting audio format: {e}")
            raise Exception(f"Failed to convert audio format: {e}")


class AudioSegmentManager:
    """Manages multiple audio segments and their metadata."""
    
    def __init__(self, audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize AudioSegmentManager.
        
        Args:
            audio_processor: AudioProcessor instance (creates new one if None)
        """
        self.audio_processor = audio_processor or AudioProcessor()
        self.segments = {}  # Dictionary to store segment metadata
        self.logger = logging.getLogger(__name__)
    
    def extract_multiple_words(
        self,
        audio_path: Union[str, Path],
        word_timestamps: list[WordTimestamp],
        output_dir: Optional[Union[str, Path]] = None,
        padding_ms: int = 100
    ) -> dict[int, str]:
        """
        Extract multiple word segments from the same audio file.
        
        Args:
            audio_path: Path to input audio file
            word_timestamps: List of WordTimestamp objects
            output_dir: Output directory (if None, uses temp directory)
            padding_ms: Additional padding in milliseconds
            
        Returns:
            Dictionary mapping word_index to output file path
        """
        try:
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix='word_segments_')
            else:
                os.makedirs(output_dir, exist_ok=True)
            
            # Load audio once
            audio_data, sample_rate = self.audio_processor.load_audio(audio_path)
            
            extracted_segments = {}
            
            for word_timestamp in word_timestamps:
                try:
                    # Extract segment
                    word_segment = self.audio_processor.extract_word_segment(
                        audio_data, sample_rate, word_timestamp, padding_ms
                    )
                    
                    # Create output filename
                    output_filename = f"word_{word_timestamp.word_index}_{word_timestamp.start_time_ms}-{word_timestamp.end_time_ms}.wav"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save segment
                    self.audio_processor.save_audio_segment(word_segment, sample_rate, output_path)
                    
                    extracted_segments[word_timestamp.word_index] = output_path
                    
                    self.logger.debug(f"Extracted word {word_timestamp.word_index} to {output_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract word {word_timestamp.word_index}: {e}")
                    continue
            
            return extracted_segments
            
        except Exception as e:
            self.logger.error(f"Error extracting multiple words: {e}")
            raise Exception(f"Failed to extract multiple word segments: {e}")
    
    def cleanup_segments(self, segment_paths: list[str]):
        """
        Clean up temporary segment files.
        
        Args:
            segment_paths: List of file paths to delete
        """
        for path in segment_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    self.logger.debug(f"Cleaned up segment: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup segment {path}: {e}")


# Convenience functions
def extract_word_from_audio(
    audio_path: Union[str, Path],
    word_timestamp: WordTimestamp,
    output_path: Optional[Union[str, Path]] = None,
    padding_ms: int = 100
) -> str:
    """
    Convenience function to extract a single word from audio.
    
    Args:
        audio_path: Path to input audio file
        word_timestamp: WordTimestamp object
        output_path: Output file path (if None, creates temporary file)
        padding_ms: Additional padding in milliseconds
        
    Returns:
        Path to extracted word audio file
    """
    processor = AudioProcessor()
    return processor.extract_and_save_word(audio_path, word_timestamp, output_path, padding_ms)


def get_audio_info(audio_path: Union[str, Path]) -> dict:
    """
    Get basic information about an audio file.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio information
    """
    processor = AudioProcessor()
    
    try:
        duration = processor.get_audio_duration(audio_path)
        audio_data, sample_rate = processor.load_audio(audio_path)
        
        return {
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "num_samples": len(audio_data),
            "file_path": str(audio_path)
        }
    except Exception as e:
        return {"error": str(e)}