"""
API clients for external Quran resources.
"""

import requests
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin


@dataclass
class WordTimestamp:
    """Represents timing information for a word in an ayah."""
    word_index: int
    start_time_ms: int
    end_time_ms: int
    segment_index: int


@dataclass
class AyahData:
    """Represents ayah data from AlQuran API."""
    surah_number: int
    ayah_number: int
    text: str
    words: List[str]
    audio_url: Optional[str] = None


@dataclass
class TarteelTimingData:
    """Represents timing data from Tarteel API."""
    surah_number: int
    ayah_number: int
    duration_ms: int
    word_timestamps: List[WordTimestamp]


class AlQuranAPIClient:
    """Client for AlQuran Cloud API."""
    
    BASE_URL = "https://api.alquran.cloud/v1/"
    
    def __init__(self):
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def get_ayah_data(
        self, 
        surah_number: int, 
        ayah_number: int = None,
        include_audio: bool = True
    ) -> AyahData:
        """
        Get ayah data from AlQuran API.
        
        Args:
            surah_number: Surah number (1-114)
            ayah_number: Ayah number (optional, if None gets full surah)
            include_audio: Whether to include Alafasy audio URL
            
        Returns:
            AyahData object containing the ayah information
        """
        try:
            if ayah_number:
                # Get specific ayah
                editions = "quran-simple-enhanced"
                if include_audio:
                    editions += ",ar.alafasy"
                
                url = f"ayah/{surah_number}:{ayah_number}/editions/{editions}"
                response = self._make_request(url)
                
                if response["code"] != 200:
                    raise Exception(f"API Error: {response.get('status', 'Unknown error')}")
                
                data = response["data"]
                
                # Extract text and audio
                text_data = next((item for item in data if item["edition"]["identifier"] == "quran-simple-enhanced"), None)
                audio_data = next((item for item in data if item["edition"]["identifier"] == "ar.alafasy"), None) if include_audio else None
                
                if not text_data:
                    raise Exception("Text data not found in API response")
                
                # Split text into words
                words = text_data["text"].split()
                
                return AyahData(
                    surah_number=surah_number,
                    ayah_number=ayah_number,
                    text=text_data["text"],
                    words=words,
                    audio_url=audio_data["audio"] if audio_data else None
                )
            else:
                # Get full surah
                editions = "quran-simple-enhanced"
                if include_audio:
                    editions += ",ar.alafasy"
                
                url = f"surah/{surah_number}/editions/{editions}"
                response = self._make_request(url)
                
                if response["code"] != 200:
                    raise Exception(f"API Error: {response.get('status', 'Unknown error')}")
                
                # For surah requests, we return the first ayah as an example
                # In practice, you might want to handle this differently
                data = response["data"]
                text_data = next((item for item in data if item["edition"]["identifier"] == "quran-simple-enhanced"), None)
                
                if not text_data or not text_data["ayahs"]:
                    raise Exception("Surah data not found in API response")
                
                first_ayah = text_data["ayahs"][0]
                words = first_ayah["text"].split()
                
                return AyahData(
                    surah_number=surah_number,
                    ayah_number=first_ayah["number"],
                    text=first_ayah["text"],
                    words=words
                )
                
        except requests.RequestException as e:
            self.logger.error(f"Network error accessing AlQuran API: {e}")
            raise Exception(f"Failed to fetch ayah data: {e}")
        except Exception as e:
            self.logger.error(f"Error processing AlQuran API response: {e}")
            raise
    
    def get_word_at_index(self, surah_number: int, ayah_number: int, word_index: int) -> str:
        """
        Get a specific word from an ayah by its index.
        
        Args:
            surah_number: Surah number
            ayah_number: Ayah number
            word_index: Zero-based word index
            
        Returns:
            The word at the specified index
        """
        ayah_data = self.get_ayah_data(surah_number, ayah_number, include_audio=False)
        
        if word_index < 0 or word_index >= len(ayah_data.words):
            raise IndexError(f"Word index {word_index} out of range for ayah with {len(ayah_data.words)} words")
        
        return ayah_data.words[word_index]
    
    def _make_request(self, endpoint: str) -> Dict:
        """Make a request to the AlQuran API."""
        url = urljoin(self.BASE_URL, endpoint)
        self.logger.debug(f"Making request to: {url}")
        
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        
        return response.json()


class TarteelAPIClient:
    """Client for Tarteel QUL API."""
    
    BASE_URL = "https://qul.tarteel.ai/resources/recitation/"
    
    def __init__(self, reciter_id: int = 334):
        """
        Initialize Tarteel API client.
        
        Args:
            reciter_id: ID of the reciter (334 for Mahmoud Khalil Al-Husary)
        """
        self.reciter_id = reciter_id
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def get_ayah_timing_data(self, surah_number: int, ayah_number: int) -> TarteelTimingData:
        """
        Get timing data for a specific ayah from Tarteel API or local segments.json file.
        
        Args:
            surah_number: Surah number (1-114)
            ayah_number: Ayah number within the surah
            
        Returns:
            TarteelTimingData object containing timing information
        """
        # First try to use local segments.json file
        try:
            import json
            import os
            
            # Look for segments.json in the project root
            segments_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "segments.json")
            
            if os.path.exists(segments_path):
                self.logger.debug(f"Using local segments.json file: {segments_path}")
                
                with open(segments_path, 'r', encoding='utf-8') as f:
                    segments_data = json.load(f)
                
                ayah_key = f"{surah_number}:{ayah_number}"
                
                if ayah_key in segments_data:
                    ayah_data = segments_data[ayah_key]
                    
                    # Extract word timestamps from segments
                    word_timestamps = []
                    if "segments" in ayah_data:
                        for segment in ayah_data["segments"]:
                            if len(segment) >= 3:
                                segment_index, start_time, end_time = segment[0], segment[1], segment[2]
                                word_timestamps.append(WordTimestamp(
                                    word_index=len(word_timestamps),  # Sequential word index
                                    start_time_ms=int(start_time),
                                    end_time_ms=int(end_time),
                                    segment_index=segment_index
                                ))
                    
                    duration_ms = ayah_data.get("duration_ms", ayah_data.get("duration_sec", 0) * 1000)
                    
                    return TarteelTimingData(
                        surah_number=surah_number,
                        ayah_number=ayah_number,
                        duration_ms=int(duration_ms),
                        word_timestamps=word_timestamps
                    )
                else:
                    self.logger.warning(f"Ayah {ayah_key} not found in local segments.json")
            
        except Exception as e:
            self.logger.warning(f"Failed to load from local segments.json: {e}")
        
        # Fallback to API (though it currently returns HTML)
        try:
            url = f"{self.BASE_URL}{self.reciter_id}"
            params = {"ayah": f"{surah_number}:{ayah_number}"}
            
            self.logger.debug(f"Making request to: {url} with params: {params}")
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Check if response is JSON
            content_type = response.headers.get('content-type', '').lower()
            if 'application/json' not in content_type and not response.text.strip().startswith('{'):
                raise Exception("API returned HTML instead of JSON - endpoint may have changed")
            
            data = response.json()
            
            # Parse the response based on the format from the web search results
            ayah_key = f"{surah_number}:{ayah_number}"
            
            if ayah_key not in data:
                raise Exception(f"Ayah {ayah_key} not found in Tarteel response")
            
            ayah_data = data[ayah_key]
            
            # Extract word timestamps from segments
            word_timestamps = []
            if "segments" in ayah_data:
                for segment in ayah_data["segments"]:
                    if len(segment) >= 3:
                        segment_index, start_time, end_time = segment[0], segment[1], segment[2]
                        word_timestamps.append(WordTimestamp(
                            word_index=len(word_timestamps),  # Sequential word index
                            start_time_ms=int(start_time),
                            end_time_ms=int(end_time),
                            segment_index=segment_index
                        ))
            
            duration_ms = ayah_data.get("duration_ms", ayah_data.get("duration_sec", 0) * 1000)
            
            return TarteelTimingData(
                surah_number=surah_number,
                ayah_number=ayah_number,
                duration_ms=int(duration_ms),
                word_timestamps=word_timestamps
            )
            
        except requests.RequestException as e:
            self.logger.error(f"Network error accessing Tarteel API: {e}")
            raise Exception(f"Failed to fetch timing data: {e}")
        except Exception as e:
            self.logger.error(f"Error processing Tarteel API response: {e}")
            raise Exception(f"Failed to fetch timing data: {e}")
    
    def get_word_timing(self, surah_number: int, ayah_number: int, word_index: int) -> WordTimestamp:
        """
        Get timing information for a specific word.
        
        Args:
            surah_number: Surah number
            ayah_number: Ayah number
            word_index: Zero-based word index
            
        Returns:
            WordTimestamp object for the specified word
        """
        timing_data = self.get_ayah_timing_data(surah_number, ayah_number)
        
        if word_index < 0 or word_index >= len(timing_data.word_timestamps):
            raise IndexError(f"Word index {word_index} out of range for ayah with {len(timing_data.word_timestamps)} word timestamps")
        
        return timing_data.word_timestamps[word_index]


# Convenience functions
def get_ayah_with_timing(surah_number: int, ayah_number: int) -> Tuple[AyahData, TarteelTimingData]:
    """
    Get both ayah data and timing information in one call.
    
    Returns:
        Tuple of (AyahData, TarteelTimingData)
    """
    alquran_client = AlQuranAPIClient()
    tarteel_client = TarteelAPIClient()
    
    ayah_data = alquran_client.get_ayah_data(surah_number, ayah_number)
    timing_data = tarteel_client.get_ayah_timing_data(surah_number, ayah_number)
    
    return ayah_data, timing_data


def get_word_info(surah_number: int, ayah_number: int, word_index: int) -> Tuple[str, WordTimestamp]:
    """
    Get word text and timing information for a specific word.
    
    Returns:
        Tuple of (word_text, WordTimestamp)
    """
    alquran_client = AlQuranAPIClient()
    tarteel_client = TarteelAPIClient()
    
    word_text = alquran_client.get_word_at_index(surah_number, ayah_number, word_index)
    word_timing = tarteel_client.get_word_timing(surah_number, ayah_number, word_index)
    
    return word_text, word_timing