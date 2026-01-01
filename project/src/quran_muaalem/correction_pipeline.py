"""
Quran recitation correction pipeline that integrates Muaalem model with external APIs.
"""

import logging
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
from pydub import AudioSegment
import requests
import time
from dotenv import load_dotenv
import re
import difflib
import bisect
import math

# --- NEW: Load environment variables from .env file ---
load_dotenv()


# Fix for LLVM SVML error on Windows
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

import numpy as np

# Import librosa with error handling
try:
    from librosa.core import load
    LIBROSA_AVAILABLE = True
except Exception as e:
    logging.warning(f"Librosa not available: {e}")
    LIBROSA_AVAILABLE = False
    def load(audio_path, sr=None, mono=True):
        raise RuntimeError("Librosa not available for audio loading")

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from quran_transcript.utils import PartOfUthmaniWord

from .inference import Muaalem
from .muaalem_typing import MuaalemOutput, Sifa
from .api_clients import AlQuranAPIClient, TarteelAPIClient, AyahData, TarteelTimingData, WordTimestamp
from .audio_processing import AudioProcessor, extract_word_from_audio
# Fixed voice prompts for error correction
INTRO_VOICE = "assets/audio/intro_voice.mp3"      # "لقد نطقت"
WRONG_VOICE = "assets/audio/wrong_voice.mp3"      # "بشكل خاطئ"
CORRECT_VOICE = "assets/audio/correct_voice.mp3"  # "والنطق الصحيح هو"




@dataclass
class ErrorDetails:
    """Represents details of a recitation error."""
    error_type: str
    description: str
    confidence: float
    expected_value: str
    actual_value: str
    phoneme_group: str


@dataclass
class WordError:
    """Represents an error at the word level."""
    surah_number: int
    ayah_number: int
    word_index: int
    word_text: str
    errors: List[ErrorDetails]
    word_audio_path: Optional[str] = None
    reference_audio_url: Optional[str] = None
    user_context_audio_path: Optional[str] = None
    reference_context_audio_path: Optional[str] = None


@dataclass
class UserWordTimestamp:
    word: str
    start_ms: int
    end_ms: int


@dataclass
class CorrectionResult:
    """Complete result of the correction pipeline."""
    surah_number: int
    ayah_number: int
    ayah_text: str
    total_words: int
    errors_found: List[WordError]
    overall_accuracy: float
    processing_time: float



class RecitationCorrector:
    """Main class for Quran recitation correction."""
    
    def __init__(
        self,
        model_name_or_path: str = "obadx/muaalem-model-v3_2",
        device: str = "cpu",
        moshaf_attributes: Optional[MoshafAttributes] = None
    ):
        """
        Initialize the recitation corrector.
        
        Args:
            model_name_or_path: Path or name of the Muaalem model
            device: Device to run the model on ('cpu' or 'cuda')
            moshaf_attributes: Moshaf configuration for phonetization
        """
        self.logger = logging.getLogger(__name__)
        
        # --- Get API key for AssemblyAI ---
        self.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.assemblyai_api_key:
            self.logger.warning("ASSEMBLYAI_API_KEY not found. User audio slicing will be disabled.")
        
        # Initialize components
        self.muaalem = Muaalem(model_name_or_path=model_name_or_path, device=device)
        self.alquran_client = AlQuranAPIClient()
        self.tarteel_client = TarteelAPIClient()
        self.audio_processor = AudioProcessor()
        
        # Default moshaf attributes
        self.moshaf_attributes = moshaf_attributes or MoshafAttributes(
            rewaya="hafs",
            madd_monfasel_len=4,
            madd_mottasel_len=4,
            madd_mottasel_waqf=4,
            madd_aared_len=4,
        )
        
        self.sampling_rate = 16000
        
        # Error detection thresholds
        self.confidence_threshold = 0.7
        self.error_thresholds = {
            'hams_or_jahr': 0.6,
            'shidda_or_rakhawa': 0.6,
            'tafkheem_or_taqeeq': 0.6,
            'itbaq': 0.6,
            'safeer': 0.6,
            'qalqla': 0.6,
            'tikraar': 0.6,
            'tafashie': 0.6,
            'istitala': 0.6,
            'ghonna': 0.6,
        }
        
    def _normalize_arabic_text(self, text: str) -> str:
            """
            Aggressively normalizes Arabic text for matching.
            Removes Diacritics, Dagger Alifs, Wasla, and unifies letter forms.
            """
            # 1. Remove standard diacritics (Fatha, Kasra, Damma, Sukun, Shadda, Tanween)
            # Range \u064B-\u0652 covers standard tashkeel
            text = re.sub(r'[\u064B-\u0652]', '', text)
            
            # 2. Remove Quranic specific diacritics
            # \u0670 is Dagger Alif (The small floating Alef in 'Ar-Rahman')
            # \u0653-\u065F are various Quranic pause marks and madd signs
            text = re.sub(r'[\u0670\u0653-\u065F]', '', text)
            
            # 3. Normalize Alef forms (Crucial step!)
            # Replaces Hamza variants (أ إ آ) AND Alef Wasla (ٱ) with bare Alef (ا)
            text = re.sub(r'[أإآٱ]', 'ا', text)
            
            # 4. Normalize Teh Marbuta to Ha
            text = re.sub(r'ة', 'ه', text)
            
            # 5. Normalize Alef Maqsura to Ya (Standardize 'ى' and 'ي')
            # Often distinct in Quran, but AssemblyAI might mix them.
            text = re.sub(r'ى', 'ي', text)
            
            # 6. Remove Tatweel (Kashida) just in case
            text = re.sub(r'ـ', '', text)
            
            return text
    
    
    def _clean_reference_data(self, surah_number: int, ayah_number: int, ayah_data: AyahData) -> AyahData:
        """
        Cleans the API data by:
        1. Removing Basmalah from the start of Verse 1 (except Surah 1).
        2. Removing Quranic stop marks (Waqf signs) and symbols.
        """
        # --- 1. Remove Basmalah (Bismillah) ---
        # Logic: If it's Verse 1 and NOT Surah Al-Fatiha (1), remove the preamble.
        if surah_number > 1 and ayah_number == 1:
            # The standard Basmalah consists of 4 words: بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
            # We check if the first word resembles "Bism"
            if len(ayah_data.words) >= 4 and "بِسْمِ" in ayah_data.words[0]:
                self.logger.info("Removing Basmalah from reference text...")
                # Remove the first 4 words from the list
                ayah_data.words = ayah_data.words[4:]
                # Remove it from the full text string as well
                basmalah_str = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
                if ayah_data.text.startswith(basmalah_str):
                    ayah_data.text = ayah_data.text.replace(basmalah_str, "", 1).strip()

        # --- 2. Remove Quranic Symbols (Waqf Marks like ۚ, ۖ, etc.) ---
        # Unicode range for Quranic symbols: \u06D6 to \u06ED
        # This includes: 
        # \u06DA (Small High Jeem - ۚ)
        # \u06D6 (Small High Sad - ۖ)
        # \u06D7 (Small High Qaf - ۗ)
        clean_words = []
        for word in ayah_data.words:
            # Remove the specific symbols from the word string
            cleaned_word = re.sub(r'[\u06D6-\u06ED]', '', word).strip()
            
            # Only keep the word if it still has characters (isn't empty)
            if cleaned_word:
                clean_words.append(cleaned_word)
        
        if len(clean_words) != len(ayah_data.words):
            self.logger.info(f"Removed {len(ayah_data.words) - len(clean_words)} non-word symbols (stop marks).")
            ayah_data.words = clean_words

        # --- 3. Merge Mottaqta'at (Disjointed Letters) ---
        # e.g. "Alif", "Lam", "Meem" -> "AlifLamMeem" (One word unit)
        # This ensures they are analyzed together and audio context isn't sliced too small.
        ayah_data.words = self._merge_mottaqtaat(ayah_data.words)

        return ayah_data
    
    def _merge_mottaqtaat(self, words: List[str]) -> List[str]:
        """
        Merges disjointed letters into single words if they appear split.
        Surah 2/3/29/30/31/32: Alif-Lam-Meem
        Surah 7: Alif-Lam-Meem-Sad
        Surah 10/11/12/14/15: Alif-Lam-Ra
        Surah 13: Alif-Lam-Meem-Ra
        Surah 19: Kaf-Ha-Ya-Ain-Sad
        Surah 20: Ta-Ha
        Surah 26/28: Ta-Sin-Meem
        Surah 27: Ta-Sin
        Surah 36: Ya-Sin
        Surah 40-46: Ha-Meem
        Surah 42: Ha-Meem Ain-Sin-Qaf
        """
        if not words: return words
        
        merged_words = []
        i = 0
        while i < len(words):
            # Check for specific patterns (Simplistic check based on standard Uthmani script split)
            # Typically "الم" is one word in most outputs, but if split: 'ال', 'م' or 'ا', 'ل', 'م'
            # Or if text is "ح م" -> "حم"
            
            # Heuristic: If we see single/two letter words that are known Mottaqta'at components at start
            w = words[i]
            
            # Explicit handling for common cases based on observation
            # Common components: 'الم', 'الر', 'طسم', 'كهيعص'
            # If they are already merged (e.g. 'الم'), fine.
            # If split: 'ا', 'ل', 'م' -> Merge
            
            # Logic: If current word + next word(s) look like decomposed Mottaqta'at
            # This requires a list of valid merged forms
            valid_forms = [
                'الم', 'الر', 'المص', 'المر', 'كهيعص', 'طه', 'طسم', 'طس', 'يس', 'ص', 'حم', 'عسق', 'ق', 'ن'
            ]
            
            # Try merging 2, 3, 4, 5 words
            merged_candidate = None
            skip_count = 0
            
            for k in range(5, 1, -1): # Try merging up to 5 words
                if i + k <= len(words):
                    candidate = "".join(words[i:i+k])
                    # Normalize for check
                    cand_norm = self._normalize_arabic_text(candidate)
                    for vf in valid_forms:
                        if self._normalize_arabic_text(vf) == cand_norm:
                            merged_candidate = candidate # Use the raw concatenated text
                            skip_count = k
                            break
                    if merged_candidate: break
            
            if merged_candidate:
                self.logger.info(f"Merged Mottaqta'at: {merged_candidate}")
                merged_words.append(merged_candidate)
                i += skip_count
            else:
                merged_words.append(w)
                i += 1
                
        return merged_words

    # Translation Map for Attributes
    ATTRIBUTE_TRANSLATIONS = {
        'hams': 'همس',
        'jahr': 'جهر',
        'shadeed': 'شدة',
        'rikhw': 'رخاوة',
        'between': 'توسط',
        'mofakham': 'مفخم',
        'moraqaq': 'مرقق',
        'low_mofakham': 'تفخيم نسبي',
        'motbaq': 'إطباق',
        'monfateh': 'انفتاح',
        'safeer': 'صفير',
        'no_safeer': 'لا صفير',
        'moqalqal': 'قلقلة',
        'not_moqalqal': 'لا قلقلة',
        'mokarar': 'تكرار',
        'not_mokarar': 'لا تكرار',
        'motafashie': 'تفشي',
        'not_motafashie': 'لا تفشي',
        'mostateel': 'استطالة',
        'not_mostateel': 'لا استطالة',
        'maghnoon': 'غنة',
        'not_maghnoon': 'لا غنة'
    }
    
        
    # --- Function to get timestamps from AssemblyAI ---
    def _get_user_audio_timestamps(self, audio_path: Union[str, Path], expected_words: List[str]) -> Optional[List[UserWordTimestamp]]:
        """
        Sends user audio to AssemblyAI to get accurate word-level timestamps,
        specifying the ARABIC language code for high accuracy.
        """
        if not self.assemblyai_api_key:
            return None
        headers = {"authorization": self.assemblyai_api_key}
        try:
            with open(audio_path, 'rb') as f:
                upload_response = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
            upload_response.raise_for_status()
            audio_url = upload_response.json()['upload_url']

            
            # We must specify the language_code for Arabic.
            transcript_request = {
                'audio_url': audio_url,
                'language_code': 'ar',  # Explicitly use the Arabic model
                'word_boost': expected_words,
                'boost_param': 'high'
            }
            transcript_response = requests.post("https://api.assemblyai.com/v2/transcript", json=transcript_request, headers=headers)
            transcript_response.raise_for_status()
            transcript_id = transcript_response.json()['id']

            while True:
                polling_response = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
                polling_response.raise_for_status()
                polling_json = polling_response.json()
                if polling_json['status'] == 'completed':
                    timestamps = []
                    for word in polling_json.get('words', []):
                        timestamps.append(
                            UserWordTimestamp(word=word['text'], start_ms=word['start'], end_ms=word['end'])
                        )
                    self.logger.info(f"Successfully retrieved {len(timestamps)} word timestamps from AssemblyAI.")
                    
                    # --- FIX: Merge timestamps to match Uthmani Expected Words ---
                    # E.g. "Ya" + "Ayyuha" -> "YaAyyuha"
                    if expected_words:
                        timestamps = self._merge_timestamps_to_match_uthmani(timestamps, expected_words)
                        
                    return timestamps
                elif polling_json['status'] == 'error':
                    raise RuntimeError(f"AssemblyAI transcription failed: {polling_json['error']}")
                self.logger.debug("Transcription in progress, waiting...")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API error getting user timestamps: {e}")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during timestamp retrieval: {e}")
            return None

    def _map_spelled_to_symbol(self, text: str) -> str:
        """Maps spelled-out Arabic letter names to their single character forms."""
        # Clean first
        text = self._normalize_arabic_text(text)
        
        # Extended mapping (including common ASR mis-spellings for Mottaqta'at)
        mapping = {
            # Common variations
            'الف': 'ا', 'ألف': 'ا', 'اليف': 'ا', 'أليف': 'ا', 
            'لام': 'ل', 'لا': 'ل',
            'ميم': 'م', 'صاد': 'ص',
            'را': 'ر', 'راء': 'ر',
            'كاف': 'ك', 'ها': 'ه', 'هاء': 'ه',
            'يا': 'ي', 'ياء': 'ي',
            'عين': 'ع', 'طا': 'ط', 'طاء': 'ط',
            'سين': 'س', 'قاف': 'ق', 'نون': 'ن',
            'حا': 'ح', 'حاء': 'ح',
            'ظا': 'ظ', 'ظاء': 'ظ',
            # Surah Maryam Specifics (Kaf-Ha-Ya-Ain-Sad)
            'كافها': 'كه', 'كافحا': 'كه', # Kaf + Ha
            'عيصى': 'عص', 'عيص': 'عص', 'عيصي': 'عص',   # Ain + Sad (approx)
            'كهيعص': 'كهيعص' # If fully recognized
        }

        # 1. Direct Exact Match
        if text in mapping:
            return mapping[text]
        
        # 'لام' often normalizes to 'لام' but let's be explicit
        if text == 'لام': return 'ل'
        
        # 2. Iterative Scan for Multiple Letters (e.g. "KafHa")
        # Sort keys by length desc to match longest first
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        
        result_symbols = []
        remaining_text = text
        found_any = False
        
        # While we can matching something at the START
        while remaining_text:
            match_found = False
            for key in sorted_keys:
                if remaining_text.startswith(key):
                    result_symbols.append(mapping[key])
                    remaining_text = remaining_text[len(key):]
                    match_found = True
                    found_any = True
                    break
            
            if not match_found:
                 # If we can't match prefix, skip one char (or abort?)
                 # Use heuristic: if we already matched something, maybe the rest is noise?
                 # Or just break.
                 break
        
        if found_any and len(result_symbols) > 0 and len(remaining_text) == 0:
            # Only return if we fully consumed the string (to avoid false positives on normal words)
             return "".join(result_symbols)

        # Fallback: if it's already a single letter, return it
        if len(text) == 1: return text
        
        # If no mapping found, return original text
        return text
    def _merge_timestamps_to_match_uthmani(
        self, 
        timestamps: List[UserWordTimestamp], 
        expected_words: List[str]
    ) -> List[UserWordTimestamp]:
        """
        Merges consecutive user timestamps if their combined text matches a single expected Uthmani word.
        Handles cases like "Ya" + "Ayyuha" (ASR) -> "YaAyyuha" (Uthmani).
        """
        if not timestamps or not expected_words:
            return timestamps

        merged: List[UserWordTimestamp] = []
        ts_idx = 0
        exp_idx = 0
        
        while ts_idx < len(timestamps):
            current_ts = timestamps[ts_idx]
            
            # If we ran out of expected words, just keep remaining timestamps as is
            if exp_idx >= len(expected_words):
                merged.append(current_ts)
                ts_idx += 1
                continue
            
            expected_norm = self._normalize_arabic_text(expected_words[exp_idx])
            current_norm = self._normalize_arabic_text(current_ts.word)
            
            # 1. Exact/Close Match?
            # If current word matches expected well enough, keep it and move both pointers
            if current_norm == expected_norm: # simple exact check first
                merged.append(current_ts)
                ts_idx += 1
                exp_idx += 1
                continue
                
            # 2. Look Ahead (Merge Check)
            # Check if merging current + next matches expected
            if ts_idx + 1 < len(timestamps):
                next_ts = timestamps[ts_idx + 1]
                combined_norm = self._normalize_arabic_text(current_ts.word + next_ts.word)
                
                # Check similarity using SequenceMatcher or strict equality
                # "Ya" + "Ayyuha" -> "YaAyyuha" vs "YaAyyuha"
                # Use fuzzy match to handle extra Alifs/Yas (e.g. YaAyyuha vs Yayyuha)
                match_ratio = difflib.SequenceMatcher(None, combined_norm, expected_norm).ratio()
                
                if combined_norm == expected_norm or match_ratio > 0.85:
                    # MERGE
                    self.logger.info(f"Merging timestamps: '{current_ts.word}' + '{next_ts.word}' -> Match '{expected_words[exp_idx]}' (Ratio: {match_ratio:.2f})")
                    # from quran_muaalem.correction_pipeline import UserWordTimestamp # Ensure class availability or reuse
                    
                    new_ts = UserWordTimestamp(
                        word=current_ts.word + next_ts.word,
                        start_ms=current_ts.start_ms,
                        end_ms=next_ts.end_ms
                    )
                    merged.append(new_ts)
                    ts_idx += 2 # Skip next
                    exp_idx += 1
                    continue
            
            # --- NEW: Mottaqta'at Merging Logic inside Timestamp Alignment ---
            # Check if current + next few words match a known Disjointed Letter sequence (e.g. Alif Lam Meem)
            merged_candidate_ts = None
            skip_count = 0
            
            valid_forms = ['الم', 'الر', 'المص', 'المر', 'كهيعص', 'طه', 'طسم', 'طس', 'يس', 'ص', 'حم', 'عسق', 'ق', 'ن']
            
            # Look up to 5 words ahead
            for k in range(5, 1, -1):
                if ts_idx + k <= len(timestamps):
                    # Get raw tokens
                    raw_tokens = [timestamps[i].word for i in range(ts_idx, ts_idx+k)]
                    
                    # 1. Try matching RAW concatenated (e.g. if they are just split letters 'ا', 'ل', 'م')
                    candidate_str = "".join(raw_tokens)
                    cand_norm = self._normalize_arabic_text(candidate_str)
                    
                    # 2. Try matching MAPPED tokens (e.g. 'ألف', 'لام', 'ميم' -> 'الم')
                    mapped_tokens = [self._map_spelled_to_symbol(t) for t in raw_tokens]
                    candidate_mapped_str = "".join(mapped_tokens)
                    cand_mapped_norm = self._normalize_arabic_text(candidate_mapped_str)
                    
                    match_found = False
                    # Check if expected word is a Mottaqta'at (e.g. 'الم') -> Enable prefix matching
                    is_expected_mottaqtaat = expected_norm in valid_forms
                    
                    for vf in valid_forms:
                        vf_norm = self._normalize_arabic_text(vf)
                        # Original Exact Match
                        if vf_norm == cand_norm or vf_norm == cand_mapped_norm:
                            match_found = True
                            break
                        
                        # NEW: Prefix Match (only if expected word IS this Mottaqta'at)
                        # This handles cases like ASR returning "Alif", "Lam" (missing Meem) for "Alif Lam Meem"
                        if is_expected_mottaqtaat and vf_norm == expected_norm:
                             if expected_norm.startswith(cand_mapped_norm) and len(cand_mapped_norm) >= 2:
                                 match_found = True
                                 self.logger.info(f"Prefix match found for Mottaqta'at: '{cand_mapped_norm}' matches start of '{expected_norm}'")
                                 break
                    
                    if match_found:
                        # Create merged timestamp
                        start_ms = timestamps[ts_idx].start_ms
                        end_ms = timestamps[ts_idx + k - 1].end_ms
                        
                        final_word = candidate_str
                        # verification check again to pick best word
                        for vf in valid_forms:
                             vf_norm = self._normalize_arabic_text(vf)
                             if vf_norm == cand_mapped_norm:
                                 final_word = "".join(mapped_tokens) # Use symbolic form
                                 break
                             # NEW: Use symbolic form if it was a prefix match
                             if is_expected_mottaqtaat and vf_norm == expected_norm and expected_norm.startswith(cand_mapped_norm) and len(cand_mapped_norm) >= 2:
                                 final_word = "".join(mapped_tokens)
                                 break
                        
                        # Check if this matches the current expected word (Uthmani) to use its text
                        if exp_idx < len(expected_words):
                             exp_word = expected_words[exp_idx]
                             exp_norm = self._normalize_arabic_text(exp_word)
                             cand_norm = self._normalize_arabic_text(final_word)
                             
                             matches = exp_norm == cand_norm
                             fuzzy = difflib.SequenceMatcher(None, cand_norm, exp_norm).ratio() > 0.8
                             prefix = (exp_norm in valid_forms or cand_norm in valid_forms) and exp_norm.startswith(cand_norm) and len(cand_norm) >= 2
                             
                             if matches or fuzzy or prefix:
                                 # Use the exact Uthmani text for better display/alignment
                                 final_word = exp_word
                                 # We will advance exp_idx later (Lines 587+)
                        
                        merged_candidate_ts = UserWordTimestamp(
                            word=final_word,
                            start_ms=start_ms,
                            end_ms=end_ms
                        )
                        skip_count = k
                        self.logger.info(f"Merging Mottaqta'at Timestamps: {raw_tokens} -> '{final_word}' ({start_ms}-{end_ms}ms)")
                        break
                        
            if merged_candidate_ts:
                merged.append(merged_candidate_ts)
                ts_idx += skip_count
                # IMPORTANT: If expected word matches this merged form, advance expected index
                # If "AlifLamMeem" is expected, we consume it.
                if exp_idx < len(expected_words):
                    exp_norm = self._normalize_arabic_text(expected_words[exp_idx])
                    cand_norm = self._normalize_arabic_text(merged_candidate_ts.word)
                    
                    matches = exp_norm == cand_norm
                    fuzzy_matches = difflib.SequenceMatcher(None, cand_norm, exp_norm).ratio() > 0.8
                    # Allow prefix match IF expected word is Mottaqta'at
                    prefix_matches = (exp_norm in valid_forms) and exp_norm.startswith(cand_norm) and len(cand_norm) >= 2
                    
                    if matches or fuzzy_matches or prefix_matches:
                        exp_idx += 1
                continue
            # ----------------------------------------------------------------
            
            # 3. Fallback
            # Neither single nor merged matched perfectly. 
            # We assume 1-to-1 mapping for now or leave misalignments to be handled later.
            merged.append(current_ts)
            ts_idx += 1
            # Only advance expected if we "think" we passed it, but sticking to 1-to-1 is safer for unmerged
            # Heuristic: if current ts is "Ya" and expected is "YaAyyuha", we technically "used" part of expected.
            # But simpler logic: just append and check next.
            # If we are failing to match, we might drift. 
            # Let's try to advance exp_idx only if it was a partial match?
            # For robustness: if we didn't merge, we don't advance exp_idx unless current matches somewhat?
            # Actually, to avoid complex drift, let's just advance both if we added one. 
            # Refinement: strict merging only.
            
            # Improve loop advancement:
            # If we didn't merge, we still consume current_ts. 
            # Should we consume exp_idx? 
            # If current_ts doesn't match expected, it might be an error or extra word.
            # Let's check similarity.
            match_ratio = difflib.SequenceMatcher(None, current_norm, expected_norm).ratio()
            if match_ratio > 0.7:
                 exp_idx += 1
            
        return merged

    

    def _align_detected_words(
        self,
        detected_timestamps: List[UserWordTimestamp],
        reference_words: List[str]
    ) -> Dict[int, Optional[UserWordTimestamp]]:
        """
        Aligns detected words (user audio) with reference Uthmani words using content-based matching.
        Returns a map: {ReferenceWordIndex -> UserTimestamp (or None)}.
        This robustly handles skipped words.
        """
        # Normalize for alignment
        det_norm = [self._normalize_arabic_text(t.word) for t in detected_timestamps]
        ref_norm = [self._normalize_arabic_text(w) for w in reference_words]
        
        matcher = difflib.SequenceMatcher(None, ref_norm, det_norm)
        
        alignment_map = {}
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Perfect match sequence
                for offset in range(i2 - i1):
                    alignment_map[i1 + offset] = detected_timestamps[j1 + offset]
            elif tag == 'replace':
                # Substitution block
                # We map whatever we have, but count matching ratio?
                # Actually, if SequenceMatcher says replace, it means "content mismatch" but "positionally aligned".
                # We will optimistically map them to allow "pronunciation correction" to inspect them.
                # However, if counts differ, we might be in trouble (e.g. 1 word replaced by 2).
                # Simple logic: map 1-to-1 as far as possible.
                count = min(i2 - i1, j2 - j1)
                for offset in range(count):
                     alignment_map[i1 + offset] = detected_timestamps[j1 + offset]
            elif tag == 'delete':
                 # Ref has words but Det does not -> SKIPPED / MISSING
                 for offset in range(i2 - i1):
                     alignment_map[i1 + offset] = None # Explicit missing
            elif tag == 'insert':
                 # Det has extra words. We can't map them to Ref indices easily.
                 # They are "Extra". We ignore them for now or log them?
                 # Current pipeline focuses on Reference Words. Extra words ideally tracked separately.
                 pass

        return alignment_map
    
    def correct_recitation(
        self, audio_path: Union[str, Path], surah_number: int, ayah_number: int,
        start_word_index: int = 0, num_words: Optional[int] = None
    ) -> CorrectionResult:
        """
        Main correction pipeline.
        
        Args:
            audio_path: Path to student's audio recitation
            surah_number: Surah number (1-114)
            ayah_number: Ayah number
            start_word_index: Starting word index (0-based)
            num_words: Number of words to analyze (if None, analyzes all words from start)
            
        Returns:
            CorrectionResult with detailed error analysis
        """
        start_time = time.time()
        
        try:
            print("\n" + "="*50)
            print("🚀 ADVANCED ANALYSIS STARTED")
            print(f"📍 Target: Surah {surah_number}, Ayah {ayah_number}")
            print("="*50)
            
            self.logger.info(f"Starting correction for Surah {surah_number}, Ayah {ayah_number}")
            
            # --- STEP 1: Get Ayah Data (LOCALLY) ---
            print("\n[Step 1] Fetching Uthmani Text (Local Library)...")
            
            # Use the local Aya class (Same as Classic Analysis)
            # This ensures we get the exact Uthmani script.
            aya_obj = Aya(surah_number, ayah_number).get()
            
            # Extract Text
            uthmani_text = aya_obj.uthmani
            
            # Extract Words
            # We split the Uthmani text by spaces to get the word list
            uthmani_words = uthmani_text.split()
            
            # Construct AyahData manually
            ayah_data = AyahData(
                surah_number=surah_number,
                ayah_number=ayah_number,
                text=uthmani_text,
                words=uthmani_words,
                audio_url=None # We don't need the API's audio (Alafasy)
            )
                        
            print(f"   - Uthmani Text: {ayah_data.text[:100]}...")
            print(f"   - Word Count: {len(ayah_data.words)}")
            print(f"   - Word List: {ayah_data.words}")
            print("-" * 50)
        
            print("\n[Step 2] Sending Audio to AssemblyAI for Alignment...")
            user_timestamps = self._get_user_audio_timestamps(audio_path, ayah_data.words)

            # --- DEBUG: Print Step 2 Results ---
            if user_timestamps:
                print(f"   - ✅ Timestamps Received: {len(user_timestamps)} segments")
                print("   - Sample Alignment (First 5 detected):")
                for i, ts in enumerate(user_timestamps[:5]):
                    print(f"     [{i}] '{ts.word}' : {ts.start_ms}ms -> {ts.end_ms}ms")
                
                if len(user_timestamps) > 5:
                    print(f"     ... (and {len(user_timestamps)-5} more)")
            else:
                print("   ⚠️ No timestamps returned (or API Key missing). Using full length fallback.")
            print("-" * 50)

            # --- START: Dynamic Word Count Logic ---
            # Default to the number of words passed from the UI, or the full Ayah.
            words_to_analyze_count = num_words
            if words_to_analyze_count is None:
                words_to_analyze_count = len(ayah_data.words) - start_word_index

            # If we have a transcript, try to find the user's stopping point using Sequence Alignment.
            if user_timestamps:
                print("\n[Step 2.1] Calculating Stopping Point...")
                # Normalize both the transcribed words and the reference words for reliable matching.
                normalized_transcribed = [self._normalize_arabic_text(ts.word) for ts in user_timestamps]
                normalized_reference = [self._normalize_arabic_text(word) for word in ayah_data.words]
                
                # Use SequenceMatcher to find the longest contiguous matching blocks.
                matcher = difflib.SequenceMatcher(None, normalized_transcribed, normalized_reference)
                matching_blocks = matcher.get_matching_blocks()
                
                # Find the index of the LAST matched word in the Reference
                last_ref_index = -1
                
                for match in matching_blocks:
                    if match.size > 0:
                        # The end index of this specific match block in the reference
                        end_of_block = match.b + match.size
                        if end_of_block > last_ref_index:
                            last_ref_index = end_of_block

                if last_ref_index != -1:
                    dynamic_count = last_ref_index
                    
                    # --- FIX: Robustness Check ---
                    # If we detected more audio segments (timestamps) than what we matched textually,
                    # it likely means the user said the words but pronounced them wrong (so match failed).
                    # We should trust the audible word count as a strong signal.
                    detected_word_count = len(user_timestamps)
                    if detected_word_count > dynamic_count:
                        print(f"   ⚠️ Text match stopped at {dynamic_count}, but detected {detected_word_count} audio segments.")
                        print(f"      Extending analysis to include potentially mispronounced words.")
                        dynamic_count = min(len(ayah_data.words), detected_word_count)

                    adjusted_count = max(0, dynamic_count - start_word_index)
                    
                    if adjusted_count > 0:
                        words_to_analyze_count = adjusted_count
                        print(f"   - 🛑 Endpoint Detected: User stopped at word index {dynamic_count} ('{ayah_data.words[dynamic_count-1] if dynamic_count > 0 else 'Start'}')")
                        print(f"   - New Word Count to Analyze: {words_to_analyze_count}")
                        self.logger.info(f"User recitation endpoint detected. Analyzing first {words_to_analyze_count} words.")
                else:
                    print("   ⚠️ Could not match audio to text. analyzing full range.")
            
            print("-" * 50)
            
            
            # Step 3: Determine the final word range for analysis
            print("\n[Step 3] Finalizing Word Range...")
            end_word_index = min(start_word_index + words_to_analyze_count, len(ayah_data.words))
            words_to_analyze = ayah_data.words[start_word_index:end_word_index]

            if not words_to_analyze:
                raise ValueError(
                    "The calculated word range for analysis is empty. "
                    "This may happen if the user's speech could not be matched to the selected Ayah."
                )

            # --- NEW: Robust Alignment for Skipped Words ---
            aligned_user_timestamps = []
            if user_timestamps:
                # 1. Align GLOBAL detected list against GLOBAL reference list (Ayah Words)
                #    We align against ayah_data.words to find the best global fit.
                global_alignment_map = self._align_detected_words(user_timestamps, ayah_data.words)
                
                # 2. Extract only the slice we are analyzing
                alignment_success_count = 0
                for i in range(len(words_to_analyze)):
                     abs_index = start_word_index + i
                     timestamp = global_alignment_map.get(abs_index)
                     aligned_user_timestamps.append(timestamp)
                     if timestamp: alignment_success_count += 1
                
                print(f"   - Aligned {alignment_success_count}/{len(words_to_analyze)} words. (Skipped: {len(words_to_analyze)-alignment_success_count})")
            else:
                aligned_user_timestamps = None
            
            # --- DEBUG INFO ---
            print(f"   - Range: Index {start_word_index} to {end_word_index}")
            print(f"   - Word Count: {len(words_to_analyze)}")
            print(f"   - Target Words: {words_to_analyze}")
            print("-" * 50)

            self.logger.info(f"Analyzing {len(words_to_analyze)} words: {' '.join(words_to_analyze)}")
            
            
            # Step 4: Get reference phonetic script
            print("\n[Step 4] Generating Reference Phonetics (The Ideal Recitation)...")
            try:
                reference_phonetic = self._get_reference_phonetic_script(
                    surah_number, ayah_number, start_word_index, words_to_analyze_count
                )
                
                # --- DEBUG INFO ---
                # reference_phonetic.text might not be available directly on the output object depending on version
                # so we rely on the prints inside the function above.
                print(f"   - Total Phoneme Groups: {len(reference_phonetic.sifat)}")
                print("-" * 50)
                
            except Exception as e:
                print(f"   ❌ Error in Step 4: {e}")
                raise e
            
            # Step 5: Process with Muaalem (Capture Duration)
            print("\n[Step 5] Process with Muaalem (Capture Duration)...")
            muaalem_output, audio_duration = self._process_audio_with_muaalem(audio_path, reference_phonetic)
            
            # Step 6: Analyze errors (Pass Duration)
            # Step 6: Analyze errors (Pass Duration)
            word_errors = self._analyze_errors(
                muaalem_output, reference_phonetic, ayah_data, aligned_user_timestamps if aligned_user_timestamps is not None else user_timestamps,
                start_word_index, len(words_to_analyze), audio_path, audio_duration
            )
            
            # Step 6: Calculate overall accuracy
            total_phonemes = len(muaalem_output.sifat)
            error_count = sum(len(error.errors) for error in word_errors)
            overall_accuracy = max(0.0, (total_phonemes - error_count) / total_phonemes) if total_phonemes > 0 else 1.0
            
            processing_time = time.time() - start_time
            
            return CorrectionResult(
                surah_number=surah_number,
                ayah_number=ayah_number,
                ayah_text=ayah_data.text,
                total_words=len(words_to_analyze),
                errors_found=word_errors,
                overall_accuracy=overall_accuracy,
                processing_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in correction pipeline: {e}")
            raise Exception(f"Correction pipeline failed: {e}")
    
    def _get_ayah_and_timing_data(self, surah_number: int, ayah_number: int) -> AyahData:
        """Get ayah data from the AlQuran API."""
        try:
            ayah_data = self.alquran_client.get_ayah_data(surah_number, ayah_number, include_audio=True)
            self.logger.debug(f"Retrieved ayah data: {len(ayah_data.words)} words")
            return ayah_data
        except Exception as e:
            self.logger.error(f"Error retrieving ayah data: {e}")
            raise
        
    
    def _get_reference_phonetic_script(self, surah_number: int, ayah_number: int, start_word: int, num_words: int):
        """
        Get reference phonetic script for the specified word range.
        Includes debug printing and handles Uthmani word boundaries.
        """
        print(f"\n   [Internal Step 4] Generating Phonetics for {num_words} words...")
        
        try:
            # Attempt to get the exact range
            verse_obj = Aya(surah_number, ayah_number)
            try:
                # Try strict slicing
                fragment_data = verse_obj.get_by_imlaey_words(start_word, num_words)
                uthmani_ref = fragment_data.uthmani
            except PartOfUthmaniWord:
                # If we cut a word in half (e.g. Ya-Ayyuha), extend by 1 to get the whole word
                print("   ⚠️ Range cuts a compound word. Extending by 1 word...")
                fragment_data = verse_obj.get_by_imlaey_words(start_word, num_words + 1)
                uthmani_ref = fragment_data.uthmani

            print(f"   - Selected Uthmani Segment: {uthmani_ref}")
            
            # Convert to phonetic script
            phonetizer_out = quran_phonetizer(
                uthmani_ref, self.moshaf_attributes, remove_spaces=True
            )
            
            print(f"   - Generated Phonemes: {phonetizer_out.phonemes}")
            self.logger.debug(f"Generated phonetic script: {phonetizer_out.phonemes}")
            return phonetizer_out
            
        except Exception as e:
            self.logger.error(f"Error generating phonetic script: {e}")
            raise
    
    def _process_audio_with_muaalem(self, audio_path: Union[str, Path], reference_phonetic) -> Tuple[MuaalemOutput, float]:
        """Process student's audio with the Muaalem model. Returns: (ModelOutput, AudioDuration)"""
        try:
            # Load audio
            wave, _ = load(audio_path, sr=self.sampling_rate, mono=True)
            duration = len(wave) / self.sampling_rate  # Calculate duration in seconds
            
            # Run Muaalem model
            outputs = self.muaalem(
                [wave],
                [reference_phonetic],
                sampling_rate=self.sampling_rate,
            )
            
            if not outputs:
                raise Exception("Muaalem model returned no output")
            
                        # =========================================================
            print("\n" + "="*50)
            print("🤖 RAW MODEL OUTPUT (First 5 Phoneme Groups):")
            print("="*50)
            
            # The output contains a list of 'sifat' (phoneme attributes)
            sifat_list = outputs[0].sifat
            
            print(f"Total Phoneme Groups Detected: {len(sifat_list)}")
            
            # Print the first 5 groups to see the structure without flooding the console
            for i, sifa in enumerate(sifat_list[:5]): 
                print(f"\n[Phoneme Group {i}]:")
                # Iterate over attributes like 'ghonna', 'qalqla', etc.
                for attr, value in sifa.__dict__.items():
                    print(f"   - {attr}: {value}")
            
            print("\n" + "="*50 + "\n")
            # =========================================================
            
            self.logger.debug(f"Muaalem processed {len(outputs[0].sifat)} phoneme groups")
            
            return outputs[0], duration
            
        except Exception as e:
            self.logger.error(f"Error processing audio with Muaalem: {e}")
            raise
    
    def _analyze_errors(
        self,
        muaalem_output: MuaalemOutput,
        reference_phonetic,
        ayah_data: AyahData,
        user_timestamps: Optional[List[UserWordTimestamp]],
        start_word_index: int,
        num_words: int,
        original_audio_path: Union[str, Path],
        audio_duration: float
    ) -> List[WordError]:
        """Perform context-aware error analysis for a word range.

        Assumes _map_phonemes_to_words returns:
        word_to_phoneme_groups: Dict[int, List[Sifa]]
        word_to_ts_idx: Dict[int, Tuple[int,int]]
        word_to_ref_phonemes: Dict[int, List[str]]
        word_to_ref_sifat: Dict[int, List[Sifa]]
        """
        print("\n[Step 5] Starting Error Analysis (Context-Aware + Fallback)...")
        word_errors: List[WordError] = []

        try:
            reference_words = [ayah_data.words[start_word_index + i] for i in range(num_words)]

            # Debug prints for incoming model output
            sifat_list = muaalem_output.sifat
            print("DEBUG: total predicted sifat items:", len(sifat_list))
            print("DEBUG: predicted phonemes groups (first 50):", [getattr(s, 'phonemes_group', '') for s in sifat_list[:50]])
            print("DEBUG: reference phonemes string:", reference_phonetic.phonemes)
            print("DEBUG: reference sifat count:", len(reference_phonetic.sifat) if hasattr(reference_phonetic, 'sifat') else 0)
            if user_timestamps:
                print("DEBUG: user timestamps count:", len(user_timestamps), "first:", user_timestamps[0].start_ms, user_timestamps[0].end_ms)

            # --- MAP AUDIO TO WORDS ---
            word_to_pred_parts, word_to_pred_sifat, word_to_ref_parts, word_to_ts_idx, word_to_ref_sifat = \
            self._map_phonemes_to_words_by_parts(
                muaalem_output.sifat,
                reference_words,
                user_timestamps,
                audio_duration,
                reference_phonetic
            )

            for word_idx in range(num_words):
                absolute_word_idx = start_word_index + word_idx
                word_text = ayah_data.words[absolute_word_idx]

                # Get mapped data (may be empty lists)
                phoneme_groups: List[Sifa] = word_to_pred_sifat.get(word_idx, [])
                ref_phonemes_list = word_to_ref_parts.get(word_idx, [])
                # Use the REAL Sifa objects now
                ref_sifat_list: List[Sifa] = word_to_ref_sifat.get(word_idx, [])

                print(f"\n   ➤ Word {word_idx + 1}: '{word_text}'")

                # Build printable representations without mutating originals
                try:
                    r_str = "".join([str(p) if not hasattr(p, 'text') else str(p.text) for p in ref_phonemes_list])
                except Exception:
                    r_str = "".join([str(x) for x in ref_phonemes_list])
                p_str = "".join([getattr(p, 'phonemes_group', '') or getattr(p, 'text', '') or '' for p in phoneme_groups])

                print(f"     ℹ️  Ref: {r_str}")
                print(f"     ℹ️  Usr: {p_str}")

                # --- HANDLE MISALIGNMENT / MISSING CASES ---
                if not phoneme_groups and not ref_phonemes_list:
                    # nothing to compare
                    print("     ⚠️ Both reference and predicted phonemes are empty for this word — skipping.")
                    continue

                if not phoneme_groups:
                    # No predicted groups mapped to this word — treat as missing alignment
                    print(f"     ⚠️ No predicted phoneme groups mapped to word index {word_idx} ('{word_text}'). Marking as missing.")
                    word_errors.append(WordError(
                        surah_number=ayah_data.surah_number,
                        ayah_number=ayah_data.ayah_number,
                        word_index=absolute_word_idx,
                        word_text=word_text,
                        errors=[ErrorDetails(
                            error_type="missing_alignment",
                            description=f"لم يتم إيجاد أصوات مُطابقة لكلمة '{word_text}' — قد يكون الانقسام غير صحيح أو لم تُنطق الكلمة بوضوح.",
                            confidence=1.0,
                            expected_value="".join(ref_phonemes_list) if ref_phonemes_list else "",
                            actual_value="",
                            phoneme_group="None"
                        )],
                        word_audio_path=None,
                        reference_audio_url=ayah_data.audio_url,
                        user_context_audio_path=None
                    ))
                    continue

                # --- COMPARE: call analyzer that checks sifat vs reference sifat/phonemes ---
                word_error_details = self._analyze_word_errors(
                    phoneme_groups,
                    ref_sifat_list,
                    ref_phonemes_list,
                    word_text
                )

                if word_error_details:
                    print(f"     ❌ Errors: {len(word_error_details)}")

                    # Try to extract contextual audio clip for this word using timestamps mapping
                    user_context_path = None
                    # prefer using the word_to_ts_idx mapping (predicted indices -> timestamps mapping)
                    ts_range = word_to_ts_idx.get(word_idx)
                    if user_timestamps and ts_range and len(user_timestamps) > 0:
                        # If user_timestamps length equals words, use its own timings, else safe-guard
                        try:
                            # If timestamp list maps 1:1 to words, we can use index word_idx
                            if len(user_timestamps) >= (word_idx + 1):
                                user_context_path = self._extract_contextual_audio_clip(
                                    str(original_audio_path),
                                    word_idx,
                                    user_timestamps
                                )
                        except Exception as e:
                            self.logger.debug(f"Failed to extract contextual clip using timestamps: {e}")
                            user_context_path = None

                    # Fallback: attempt to extract using the start/end ratio of predicted indices if possible
                    if not user_context_path:
                        try:
                            # If we have a reasonable ts_range (predicted indices), try to map to time using audio_duration
                            # but only if user_timestamps are not available or failed.
                            if audio_duration and audio_duration > 0 and ts_range:
                                pred_start_idx, pred_end_idx = ts_range
                                total_preds = len(sifat_list) if sifat_list else 0
                                if total_preds > 0:
                                    start_ratio = pred_start_idx / total_preds
                                    end_ratio = pred_end_idx / total_preds if pred_end_idx > pred_start_idx else min(1.0, (pred_start_idx + 1) / total_preds)
                                    est_start_ms = int(start_ratio * audio_duration * 1000)
                                    est_end_ms = int(end_ratio * audio_duration * 1000)
                                    # safe-check
                                    if est_end_ms > est_start_ms:
                                        try:
                                            full_audio = AudioSegment.from_file(str(original_audio_path))
                                            clip = full_audio[est_start_ms:est_end_ms]
                                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                                                clip.export(fp.name, format="mp3")
                                                user_context_path = fp.name
                                        except Exception as e:
                                            self.logger.debug(f"Failed fallback audio slice by estimated ms: {e}")
                                            user_context_path = None
                        except Exception as e:
                            self.logger.debug(f"Error while attempting fallback audio extraction: {e}")
                            user_context_path = None

                    if not user_context_path:
                        user_context_path = str(original_audio_path)

                    word_errors.append(WordError(
                        surah_number=ayah_data.surah_number,
                        ayah_number=ayah_data.ayah_number,
                        word_index=absolute_word_idx,
                        word_text=word_text,
                        errors=word_error_details,
                        word_audio_path=user_context_path,
                        reference_audio_url=ayah_data.audio_url,
                        user_context_audio_path=user_context_path,
                    ))
                else:
                    print("     ✅ Correct")

            return word_errors

        except Exception as e:
            self.logger.error(f"Error analyzing errors: {e}", exc_info=True)
            raise

    
    def _word_has_timestamp(self, word_rel_idx, timestamps, ref_words):
        """Helper to check if we actually mapped a timestamp to this word index."""
        # Simplified: if we are here, we likely have it, handled by the upper logic
        return True
    
    def _build_predicted_concat_and_cumlen(self, predicted_sifat: List[Any]):
        """
        Returns (predicted_concat_str, predicted_cumlen_list)
        predicted_cumlen is length len(predicted_sifat)+1 where predicted_cumlen[i]
        هو عدد الأحرف قبل العنصر i في predicted_sifat.
        """
        parts = []
        for s in predicted_sifat:
            # تأكد من أننا نحصل على التمثيل النصي الصحيح لكل عنصر
            part = getattr(s, 'phonemes_group', '') or getattr(s, 'text', '') or ''
            # keep as-is (no cleaning that removes phonetic symbols)
            parts.append(str(part))

        concat = "".join(parts)
        cumlen = [0]
        for p in parts:
            cumlen.append(cumlen[-1] + len(p))
        return concat, cumlen


    # CHANGE 2: Smarter Mapping with Alignment
    def _map_phonemes_to_words_by_parts(
        self,
        predicted_sifat: List[Sifa],
        reference_words: List[str],
        user_timestamps: Optional[List[UserWordTimestamp]],
        audio_duration: float,
        full_reference_phonetic: Any
    ):
        """
        Fast, deterministic mapping:
        1) Build predicted_parts (strings) from predicted_sifat.
        2) Use the lengths of predicted_parts to slice reference_concat (the string from step 4)
        into reference_parts of matching lengths (primary method).
        - If total lengths differ, fallback to proportional distribution with clamping.
        3) Compute char boundaries for each reference word using quran_phonetizer(word).
        4) Assign each part (pred & ref) to the word whose char-boundary contains the part's midpoint.
        Returns:
        word_to_pred_parts: Dict[word_idx -> List[str]]
        word_to_pred_sifat: Dict[word_idx -> List[Sifa]]
        word_to_ref_parts: Dict[word_idx -> List[str]]
        word_to_ts_idx: Dict[word_idx -> Tuple[int,int]]  # indices into predicted_sifat
        """
        # Prepare outputs
        word_to_pred_parts = {}
        word_to_pred_sifat = {}
        word_to_ref_parts = {}
        word_to_ts_idx = {}
        word_to_ref_sifat = {} # Ensure this is initialized

        # 0) PRIORITY: If aligned timestamps are provided, use them for robust mapping
        if user_timestamps:
             total_duration_ms = audio_duration * 1000
             if total_duration_ms <= 0: total_duration_ms = 1
             total_sifat = len(predicted_sifat)
             
             # Also prepare Reference Sifat mapping (Sequential)
             ref_sifat_cursor = 0
             all_ref_sifat = getattr(full_reference_phonetic, 'sifat', [])
             
             for wi, word in enumerate(reference_words):
                 # 1. Prediction Mapping (Time-based)
                 ts = user_timestamps[wi] if wi < len(user_timestamps) else None
                 
                 word_to_pred_parts[wi] = []
                 word_to_pred_sifat[wi] = []
                 
                 if ts:
                     # Linear Time Mapping
                     s_idx = int((ts.start_ms / total_duration_ms) * total_sifat)
                     e_idx = int((ts.end_ms / total_duration_ms) * total_sifat)
                     # Clamp constraints
                     s_idx = max(0, min(s_idx, total_sifat))
                     e_idx = max(s_idx, min(e_idx, total_sifat)) # e_idx >= s_idx
                     
                     subset = predicted_sifat[s_idx:e_idx]
                     word_to_pred_sifat[wi] = subset
                     word_to_pred_parts[wi] = [getattr(s, 'phonemes_group', '') or getattr(s, 'text', '') or '' for s in subset]
                     word_to_ts_idx[wi] = (s_idx, e_idx)
                 else:
                     # Missing Word -> Empty
                     word_to_ts_idx[wi] = (0, 0)
                 
                 # 2. Reference Mapping (Sequential Char Length)
                 # Generate ISO phonemes for this word to know its length in Sifat
                 iso = quran_phonetizer(word, self.moshaf_attributes, remove_spaces=True)
                 word_iso_str = iso.phonemes
                 word_to_ref_parts[wi] = [word_iso_str]
                 
                 # Assign Ref Sifat (Assume they match 1-to-1 with phonemes generally)
                 # Heuristic: Count chars in word_iso_str, consume that many Sifat?
                 # Actually Sifat objects usually map to phoneme groups.
                 # Let's count how many Sifat objects cover 'word_iso_str'.
                 # We loop through all_ref_sifat from cursor
                 current_ref_sifat = []
                 consumed_chars = 0
                 target_len = len(word_iso_str)
                 
                 while ref_sifat_cursor < len(all_ref_sifat) and consumed_chars < target_len:
                     s = all_ref_sifat[ref_sifat_cursor]
                     s_text = getattr(s, 'phonemes_group', '') or getattr(s, 'text', '') or str(s)
                     current_ref_sifat.append(s)
                     consumed_chars += len(s_text)
                     ref_sifat_cursor += 1
                 
                 word_to_ref_sifat[wi] = current_ref_sifat
            
             return word_to_pred_parts, word_to_pred_sifat, word_to_ref_parts, word_to_ts_idx, word_to_ref_sifat

        # 1) predicted parts & concat
        predicted_parts = [ (getattr(s, 'phonemes_group', '') or getattr(s, 'text', '') or '') for s in predicted_sifat ]
        predicted_concat = "".join(predicted_parts)
        predicted_cumlen = [0]
        for p in predicted_parts:
            predicted_cumlen.append(predicted_cumlen[-1] + len(p))
        total_pred_chars = len(predicted_concat)

        # 2) reference concat (from step 4 object) and attempt to split by predicted_parts lengths
        ref_concat = full_reference_phonetic.phonemes
        total_ref_chars = len(ref_concat)

        reference_parts = []
        # If total lengths equal, do direct slicing
        sum_pred_lens = sum(len(p) for p in predicted_parts)
        if sum_pred_lens == total_ref_chars and total_ref_chars > 0:
            # exact simple split (fast)
            idx = 0
            for p in predicted_parts:
                L = len(p)
                reference_parts.append(ref_concat[idx: idx + L])
                idx += L
        else:
            # fallback: proportional slicing: allocate to each predicted part a chunk proportional to its length
            # but ensure integer lengths and total coverage of ref_concat
            if total_ref_chars == 0:
                # nothing to split; create empty placeholders
                reference_parts = [''] * len(predicted_parts)
            else:
                # compute float proportions
                pred_lens = [len(p) for p in predicted_parts]
                total_pred_lens = sum(pred_lens) or 1
                # first assign float sizes, then round while preserving total by distributing remainder
                float_sizes = [ (l / total_pred_lens) * total_ref_chars for l in pred_lens ]
                int_sizes = [math.floor(sz) for sz in float_sizes]
                remainder = total_ref_chars - sum(int_sizes)
                # distribute remainder to largest fractional parts
                fractional = [ (float_sizes[i] - int_sizes[i], i) for i in range(len(int_sizes)) ]
                fractional.sort(reverse=True)
                for frac, idx_frac in fractional[:remainder]:
                    int_sizes[idx_frac] += 1
                # now slice according to int_sizes
                pos = 0
                for L in int_sizes:
                    reference_parts.append(ref_concat[pos: pos + L])
                    pos += L
                # if any leftover (rounding), append to last
                if pos < total_ref_chars:
                    reference_parts[-1] += ref_concat[pos: total_ref_chars]

        # Safety: ensure same number of parts as predicted_parts
        if len(reference_parts) != len(predicted_parts):
            # try to re-balance simple: pad/refine
            # easiest: if reference_parts shorter, append empty strings
            while len(reference_parts) < len(predicted_parts):
                reference_parts.append('')
            # if longer, merge trailing
            if len(reference_parts) > len(predicted_parts):
                # merge extra into last
                extras = reference_parts[len(predicted_parts):]
                reference_parts = reference_parts[:len(predicted_parts)]
                reference_parts[-1] = reference_parts[-1] + "".join(extras)

        # 3) compute ref_word boundaries (char indices) using quran_phonetizer per word
        ref_word_char_boundaries = []
        cursor = 0
        for w in reference_words:
            iso = quran_phonetizer(w, self.moshaf_attributes, remove_spaces=True)
            iso_str = iso.phonemes
            # try to find iso_str in ref_concat starting from cursor
            found = ref_concat.find(iso_str, cursor)
            if found != -1:
                start_idx = found
                end_idx = found + len(iso_str)
            else:
                # fallback: assume contiguous (next chunk)
                start_idx = cursor
                end_idx = min(len(ref_concat), cursor + len(iso_str))
            ref_word_char_boundaries.append((start_idx, end_idx))
            cursor = end_idx

        # 4) assign each part to a word by midpoint char index in ref_concat
        # initialize structures
        for wi in range(len(reference_words)):
            word_to_pred_parts[wi] = []
            word_to_pred_sifat[wi] = []
            word_to_ref_parts[wi] = []

        # For predicted parts: determine char span in predicted_concat, map to equivalent char span in ref_concat
        # We'll map each predicted part i -> reference_parts[i], then use midpoint index in ref_concat of that ref_part to choose word.
        for i, (pred_part, ref_part) in enumerate(zip(predicted_parts, reference_parts)):
            # compute midpoint char index into ref_concat for this ref_part
            # need to find where this ref_part sits inside ref_concat: search (first occurrence from left)
            # to be robust, search occurrences and pick the one nearest previous cursor
            # but simpler: we can compute cumulative lengths of reference_parts to get position
            # Build cumulative of reference_parts once
            pass

        # build cumulative positions of reference_parts in ref_concat (deterministic)
        ref_parts_cum = [0]
        for rp in reference_parts:
            ref_parts_cum.append(ref_parts_cum[-1] + len(rp))
        # Now assign: ref_part i occupies ref_concat[ref_parts_cum[i]: ref_parts_cum[i+1]]
        for i, (pred_part, ref_part) in enumerate(zip(predicted_parts, reference_parts)):
            start_ref_char = ref_parts_cum[i]
            end_ref_char = ref_parts_cum[i+1]
            mid_ref = (start_ref_char + end_ref_char) // 2 if end_ref_char > start_ref_char else start_ref_char
            # find word whose boundary contains mid_ref
            assigned_word = None
            for wi, (rs, re) in enumerate(ref_word_char_boundaries):
                if rs <= mid_ref < re or (wi == len(ref_word_char_boundaries)-1 and mid_ref >= rs):
                    assigned_word = wi
                    break
            if assigned_word is None:
                assigned_word = 0
            # append string parts and Sifa objects
            word_to_ref_parts[assigned_word].append(ref_part)
            word_to_pred_parts[assigned_word].append(pred_part)
            word_to_pred_sifat[assigned_word].append(predicted_sifat[i])

        # 5) build word_to_ts_idx (indices in predicted_sifat): get min/max indices assigned
        for wi in range(len(reference_words)):
            sifa_list = word_to_pred_sifat[wi]
            if not sifa_list:
                word_to_ts_idx[wi] = (0, 0)
            else:
                # determine indices by scanning predicted_sifat and matching object identity
                indices = []
                for j, s in enumerate(predicted_sifat):
                    if s in sifa_list:
                        indices.append(j)
                if indices:
                    word_to_ts_idx[wi] = (min(indices), max(indices) + 1)
                else:
                    word_to_ts_idx[wi] = (0, 0)
        
        # 6) Map REFERENCE SIFAT objects to words
        # Iterate over full_reference_phonetic.sifat and assign to word based on mid-point in ref_concat
        word_to_ref_sifat = {wi: [] for wi in range(len(reference_words))}
        
        if hasattr(full_reference_phonetic, 'sifat') and full_reference_phonetic.sifat:
            cursor = 0
            for sifa in full_reference_phonetic.sifat:
                # determine text length of this Sifa
                s_text = getattr(sifa, 'phonemes_group', '') or getattr(sifa, 'text', '') or str(sifa)
                L = len(s_text)
                if L == 0: 
                    # fallback if empty, assume 0 advancement or skip?
                    # let's assume it doesn't advance cursor significantly or is handled
                    pass
                
                mid_ref = cursor + (L // 2)
                
                # Find which word covers this char position
                assigned_word = None
                for wi, (rs, re) in enumerate(ref_word_char_boundaries):
                    if rs <= mid_ref < re or (wi == len(ref_word_char_boundaries)-1 and mid_ref >= rs):
                        assigned_word = wi
                        break
                
                if assigned_word is not None:
                    word_to_ref_sifat[assigned_word].append(sifa)
                
                cursor += L

        return word_to_pred_parts, word_to_pred_sifat, word_to_ref_parts, word_to_ts_idx, word_to_ref_sifat


    
    def _clean_text_for_comparison(self, item, is_ref=False):
        """Extracts the base character for alignment (Removing Tashkeel/Madd temporarily)."""
        if is_ref:
            raw = item.text if hasattr(item, 'text') else str(item)
        else:
            raw = getattr(item, 'phonemes_group', '')
            
        # Clean: Remove spaces, keep only letters for structural alignment
        # We normalize 'ۦ' to 'ي' etc. to align correctly
        # UPDATE: Include Arabic Diacritics (\u064B-\u0652) so they are NOT stripped by [^\w]
        c = re.sub(r'[^\w\u064B-\u0652]', '', raw)
        c = c.replace('ۦ', 'ي').replace('ۥ', 'و').replace('ٱ', 'ا').replace('ى', 'ي')
        # Remove vowels/shadda for skeleton alignment
        # c = re.sub(r'[ًٌٍَُِّْ]', '', c) 
        # UPDATE: We want to KEEP vowels so they participate in alignment (e.g. Damma vs Kasra)
        # This allows difflib to see them as distinct tokens.
        return c

    def _clean_but_keep_vowels(self, text: str) -> str:
        """
        Clean text but PRESERVE vowels (Fatha, Damma, Kasra) for strict comparison.
        Normalizes structural chars similar to _clean_text_for_comparison.
        """
        # Normalize structural chars
        c = text.replace('ۦ', 'ي').replace('ۥ', 'و').replace('ٱ', 'ا').replace('ى', 'ي')
        
        # Remove non-word chars but KEEP Arabic Diacritics ranges
        # Arabic Block: \u0600-\u06FF
        # We want to remove special symbols but keep letters + vowels.
        # Ideally, just remove punctuation.
        
        # Keep: letters, Fatha(064E), Damma(064F), Kasra(0650), Shadda(0651), Sukun(0652), 
        # Tanween Fath(064B), Tanween Damm(064C), Tanween Kasr(064D)
        
        # Remove: Spaces, non-arabic punctuation
        c = re.sub(r'[^\w\u064B-\u0652]', '', c)
        
        # Remove some "Tatweel" or other noise if needed
        c = c.replace('_', '')
        
        return c

    def _analyze_word_errors(
        self,
        predicted_sifat: List[Sifa],
        reference_sifat: List[Sifa],
        reference_phonemes: Any,  # Can be string or list
        word_text: str
    ) -> List[ErrorDetails]:
        """
        Smart Error Analyzer using Sequence Alignment.
        Compares reference_sifat (the golden Sifa objects) to predicted_sifat (model output)
        in a robust way that handles unequal lengths, insertions, deletions, and replacements.
        Returns a list of ErrorDetails describing detected mismatches.
        """
        errors: List[ErrorDetails] = []
        
        def _pred_slice_summary(j1, j2):
            if j1 >= j2:
                return ""
            parts = []
            for idx in range(j1, min(j2, len(predicted_sifat))):
                p = predicted_sifat[idx]
                # Safely get text representation
                raw = getattr(p, "phonemes_group", "") or getattr(p, "text", "") or ""
                parts.append(raw)
            return "".join(parts)

        # Helper: get canonical token for a sifat object (keep phonetic marks; only strip whitespace)
        def _unit_key_from_sifa(s):
            try:
                if hasattr(s, "text") and s.text:
                    return self._clean_text_for_comparison(s, is_ref=True)
                # fallback to phonemes_group if available
                if hasattr(s, "phonemes_group") and getattr(s, "phonemes_group"):
                    return self._clean_text_for_comparison(getattr(s, "phonemes_group"), is_ref=True)
                return str(s)
            except Exception:
                return str(s)

        # Build ref_units from reference_sifat if available (each ref_sifat corresponds to a base unit)
        ref_units = []
        if reference_sifat:
            for r in reference_sifat:
                # prefer .text if present, otherwise try attribute that holds base letter
                if hasattr(r, 'text') and r.text:
                    ref_units.append(self._clean_text_for_comparison(r.text, is_ref=True))
                elif hasattr(r, 'phonemes') and r.phonemes:
                    ref_units.append(self._clean_text_for_comparison(r.phonemes, is_ref=True))
                elif hasattr(r, 'phonemes_group') and r.phonemes_group:
                    ref_units.append(self._clean_text_for_comparison(r.phonemes_group, is_ref=True))
                else:
                    # if r is an object with attributes representing base letter, try to stringify it
                    ref_units.append(self._clean_text_for_comparison(str(r), is_ref=True))
        else:
            # fallback to previous char-level method (only if no reference_sifat)
            ref_raw_text = reference_phonemes if isinstance(reference_phonemes, str) else "".join([str(x) for x in reference_phonemes])
            ref_units = [self._clean_text_for_comparison(ch, is_ref=True) for ch in ref_raw_text if self._clean_text_for_comparison(ch, is_ref=True)]


        # Build predicted unit list
        pred_units = [self._clean_text_for_comparison(p, is_ref=False) for p in predicted_sifat]

        # Sequence alignment
        matcher = difflib.SequenceMatcher(None, ref_units, pred_units)
        opcodes = matcher.get_opcodes()

        # Character -> readable name map (used for phoneme_group in output)
        CHAR_MAP = {
            'ء': 'همزة', 'ب': 'باء', 'ت': 'تاء', 'ث': 'ثاء', 'ج': 'جيم', 'ح': 'حاء',
            'خ': 'خاء', 'د': 'دال', 'ذ': 'ذال', 'ر': 'راء', 'ز': 'زاي', 'س': 'سين',
            'ش': 'شين', 'ص': 'صاد', 'ض': 'ضاد', 'ط': 'طاء', 'ظ': 'ظاء', 'ع': 'عين',
            'غ': 'غين', 'ف': 'فاء', 'ق': 'قاف', 'ك': 'كاف', 'ل': 'لام', 'م': 'ميم',
            'ن': 'نون', 'ه': 'هاء', 'و': 'واو', 'ي': 'ياء', 'ا': 'ألف',
            'ۦ': 'مد ياء', 'ۥ': 'مد واو', '~': 'علامة مد',
            'ى': 'ألف مقصورة', 'ة': 'تاء مربوطة', 'ٱ': 'همزة وصل'
        }

        # Utility to build readable phoneme_group summary for a predicted slice
        def _pred_slice_summary(j1, j2):
            if j1 >= j2:
                return ""
            parts = []
            for idx in range(j1, min(j2, len(predicted_sifat))):
                p = predicted_sifat[idx]
                parts.append(getattr(p, "phonemes_group", "") or getattr(p, "text", "") or str(p))
            return "".join(parts)

        # Iterate opcodes and generate errors
        for tag, i1, i2, j1, j2 in opcodes:

            # --- NEW: Check for Harakat/Vowel mismatch in "equal" blocks ---
            if tag == "equal":
                # For equal blocks, we also want to check if the actual *vowels* match.
                # The alignment (difflib) used _clean_text_for_comparison which strips vowels.
                # Now we want to be stricter.
                
                for ref_idx, pred_idx in zip(range(i1, i2), range(j1, j2)):
                    if pred_idx >= len(predicted_sifat) or ref_idx >= len(reference_sifat):
                        continue
                        
                    ref_obj = reference_sifat[ref_idx]
                    pred_obj = predicted_sifat[pred_idx]
                    
                    try:
                        # Vowel mapping for objects that might return English descriptions (e.g. "damma")
                        VOWEL_MAP = {
                            'fatha': 'َ', 'damma': 'ُ', 'kasra': 'ِ', 
                            'sukun': 'ْ', 'shadda': 'ّ', 'tanween_fath': 'ً', 
                            'tanween_damm': 'ٌ', 'tanween_kasr': 'ٍ'
                        }
                        
                        def _extract_raw(obj):
                            # Try standard attributes
                            val = getattr(obj, 'phonemes', '') or getattr(obj, 'text', '') or getattr(obj, 'phonemes_group', '') or getattr(obj, 'name', '') or str(obj)
                            # If it looks like a vowel name, map it
                            if str(val).lower() in VOWEL_MAP:
                                return VOWEL_MAP[str(val).lower()]
                            return val

                        p_raw = _extract_raw(pred_obj)
                        r_raw = _extract_raw(ref_obj)
                        
                        # Clean but preserve vowels
                        p_clean = self._clean_but_keep_vowels(p_raw)
                        r_clean = self._clean_but_keep_vowels(r_raw)
                        
                        # Debug Log for Harakat Check
                        # self.logger.debug(f"Harakat Check: Ref '{r_clean}' vs Pred '{p_clean}'")

                        # If meaningful content differs
                        if (p_clean or r_clean) and p_clean != r_clean:
                            # Verify it isn't just a harmless difference (like just different madd length representations if normalized)
                            # But here we want to catch diacritics.
                            
                            # Simple heuristic: if they differ and are short (likely single letter+vowel), flag it.
                            # We combine a confidence heuristic.
                            
                            errors.append(ErrorDetails(
                                error_type="harakat_error",
                                description=f"خطأ في التشكيل (الحركات): المتوقع '{r_clean}' والمسموع '{p_clean}'",
                                confidence=0.9, # High confidence for direct mismatch
                                expected_value=r_clean,
                                actual_value=p_clean,
                                phoneme_group=CHAR_MAP.get(self._clean_text_for_comparison(r_clean, True), r_clean)
                            ))
                            
                        # --- NEW: Check Sifat Attributes (Safeer, Hams, etc.) ---
                        # Iterate through defined thresholds and check if attributes match
                        for attr_name, threshold in self.error_thresholds.items():
                            try:
                                # Safe access to attribute
                                if not hasattr(ref_obj, attr_name) or not hasattr(pred_obj, attr_name):
                                    continue
                                
                                ref_attr = getattr(ref_obj, attr_name)
                                pred_attr = getattr(pred_obj, attr_name)
                                
                                if not ref_attr or not pred_attr:
                                    continue
                                    
                                r_val = getattr(ref_attr, 'text', None)
                                p_val = getattr(pred_attr, 'text', None)
                                p_prob = getattr(pred_attr, 'prob', 0.0)
                                if isinstance(p_prob, list): p_prob = p_prob[0] if p_prob else 0.0
                                
                                # DEBUG:
                                # if attr_name == 'safeer':
                                #     print(f"DEBUG: Safeer Check - Word '{word_text}' - Ref: {r_val} vs Pred: {p_val} (Prob: {p_prob:.2f})")

                                # Special Case: Thal (ذ) needs sensitive Safeer check
                                # Ref Char should be 'ذ' (Thal) -> no_safeer
                                # But if Pred is 'Zay' (Safeer), we want to catch it even if model is uncertain.
                                is_thal_ref = (CHAR_MAP.get(self._clean_text_for_comparison(r_clean, True), '') == 'ذال') or r_clean == 'ذ'
                                
                                if attr_name == 'safeer' and is_thal_ref:
                                    # If reference is Thal, any High Probability Safeer is an error
                                    if p_val == 'safeer' and p_prob > 0.4: # Lower threshold for this specific error
                                         errors.append(ErrorDetails(
                                            error_type='safeer',
                                            description=f"في كلمة '{word_text}': تم نطق حرف (زاي - صفير) بدلاً من (ذال)",
                                            confidence=float(p_prob),
                                            expected_value=self.ATTRIBUTE_TRANSLATIONS.get('no_safeer', 'no_safeer'),
                                            actual_value=self.ATTRIBUTE_TRANSLATIONS.get('safeer', 'safeer'),
                                            phoneme_group=p_clean or "?"
                                        ))
                                         continue # Skip standard check to avoid duplicate

                                # Standard Check
                                if r_val and p_val and r_val != p_val:
                                    if p_prob >= threshold:
                                        # Generate readable error
                                        err_desc = self._generate_error_description(attr_name, p_val, r_val, word_text)
                                        
                                        # Translate for object fields too
                                        t_expected = self.ATTRIBUTE_TRANSLATIONS.get(r_val, r_val)
                                        t_actual = self.ATTRIBUTE_TRANSLATIONS.get(p_val, p_val)
                                        
                                        errors.append(ErrorDetails(
                                            error_type=attr_name,
                                            description=err_desc,
                                            confidence=float(p_prob),
                                            expected_value=t_expected,
                                            actual_value=t_actual,
                                            phoneme_group=p_clean or "?"
                                        ))
                            except Exception as ex:
                                pass

                    except Exception as e:
                       pass

                # The prompt asks for REPLACE block logic. The equal block logic is above it.)
                # I must be careful not to overwrite the equal block if I don't intend to.
                # Actually, I am targeting lines 1274+ which is `elif tag == "replace":`.
                # But `_pred_slice_summary` is defined recursively or before?
                # Ah, _pred_slice_summary usage is inside. Definition was seemingly inside too.
                # Let's just fix the usage site logic if definition is elsewhere or rewrite inline if needed.
                pass 
                
            elif tag == "replace":
                # reference[i1:i2] replaced by predicted[j1:j2]
                ref_len = i2 - i1
                pred_len = j2 - j1
                
                # If lengths equal, map one-to-one
                if ref_len == pred_len and ref_len > 0:
                    for offset in range(ref_len):
                        ref_idx = i1 + offset
                        pred_idx = j1 + offset
                        if ref_idx >= len(reference_sifat) or pred_idx >= len(predicted_sifat):
                            continue
                        ref_obj = reference_sifat[ref_idx]
                        pred_obj = predicted_sifat[pred_idx]
                        
                        # treat as articulation mismatch (substitution)
                        base_char = ref_units[ref_idx] if ref_idx < len(ref_units) else "?"
                        
                        # Get RAW text for display (with Tashkeel)
                        exp_display = getattr(ref_obj, 'phonemes', '') or getattr(ref_obj, 'text', '') or getattr(ref_obj, 'phonemes_group', '') or str(ref_obj)
                        
                        try:
                            # Re-use extract raw (should be helper method but defining inline is safer for scope)
                            VOWEL_MAP = {
                                'fatha': 'َ', 'damma': 'ُ', 'kasra': 'ِ', 
                                'sukun': 'ْ', 'shadda': 'ّ', 'tanween_fath': 'ً', 
                                'tanween_damm': 'ٌ', 'tanween_kasr': 'ٍ'
                            }
                            def _extract_raw(obj):
                                val = getattr(obj, 'phonemes_group', '') or getattr(obj, 'text', '') or getattr(obj, 'phonemes', '') or str(obj)
                                if str(val).lower() in VOWEL_MAP:
                                    return VOWEL_MAP[str(val).lower()]
                                return val

                            act_base = self._clean_text_for_comparison(pred_obj, is_ref=False)
                            # If cleaning yields empty (just vowel?), try raw
                            act_name = CHAR_MAP.get(act_base, act_base)
                            
                            # If CHAR_MAP returns empty or symbol, try fetching the raw vowel
                            p_raw_v = _extract_raw(pred_obj)
                            if not act_base and p_raw_v:
                                act_name = p_raw_v
                            elif p_raw_v and p_raw_v not in act_name:
                                # Append vowel if missing? 
                                # Actually we just want the raw display if it's vastly different
                                act_name = p_raw_v
                                
                        except Exception:
                            # Fallback but avoid dumping object
                            act_name = getattr(pred_obj, "phonemes_group", "") or "?"
                            
                        errors.append(ErrorDetails(
                            error_type="articulation",
                            description=f"في كلمة '{word_text}': إبدال حرف - نُطق '{act_name}' بدلاً من '{exp_display}'",
                            confidence=1.0,
                            expected_value=exp_display,
                            actual_value=act_name,
                            phoneme_group=base_char
                        ))
                else:
                    # lengths unequal: create proportional mapping or mark whole block as substitution
                    # Build summaries using SAFE access
                    ref_summary = "".join([ (getattr(r, "phonemes_group", "") or getattr(r, "text", "") or "") for r in reference_sifat[i1:i2] ])
                    pred_summary = _pred_slice_summary(j1, j2)
                    
                    errors.append(ErrorDetails(
                        error_type="articulation_block",
                        description=f"في كلمة '{word_text}': اختلاف في مجموعة الحروف المرجعية مقابل المنطوقة. المرجع='{ref_summary}'، المنطوق='{pred_summary}'",
                        confidence=1.0,
                        expected_value=ref_summary,
                        actual_value=pred_summary,
                        phoneme_group=ref_units[i1] if i1 < len(ref_units) else ""
                    ))

            elif tag == "delete":
                # reference segment deleted in prediction -> missing pronunciation(s)
                for ref_idx in range(i1, i2):
                    base_char = ref_units[ref_idx] if ref_idx < len(ref_units) else "?"
                    exp_name = CHAR_MAP.get(base_char, base_char)
                    errors.append(ErrorDetails(
                        error_type="articulation",
                        description=f"في كلمة '{word_text}': حذف حرف - لم يتم نطق '{exp_name}'",
                        confidence=1.0,
                        expected_value=exp_name,
                        actual_value="محذوف",
                        phoneme_group=base_char
                    ))

            elif tag == "insert":
                # predicted inserted extra units not present in reference -> extra pronunciation
                # summarize inserted predicted slice
                pred_summary = _pred_slice_summary(j1, j2)
                errors.append(ErrorDetails(
                    error_type="extra_pronunciation",
                    description=f"في كلمة '{word_text}': نطق إضافي لمجموعة أصوات غير متوقعة: '{pred_summary}'",
                    confidence=1.0,
                    expected_value="",
                    actual_value=pred_summary,
                    phoneme_group=pred_summary[:1] if pred_summary else ""
                ))

        return errors

    def _generate_error_description(self, error_type: str, actual: str, expected: str, word_text: str) -> str:
        """Generate human-readable error descriptions in clean Arabic."""
        
        # Translate values
        t_actual = self.ATTRIBUTE_TRANSLATIONS.get(actual, actual)
        t_expected = self.ATTRIBUTE_TRANSLATIONS.get(expected, expected)
        t_type = error_type # Can be translated if needed, but usually redundant in the message
        
        # Attribute specific maps
        ar_types = {
            'hams_or_jahr': 'الهمس والجهر',
            'shidda_or_rakhawa': 'الشدة والرخاوة',
            'tafkheem_or_taqeeq': 'التفخيم والترقيق',
            'ghonna': 'الغنة',
            'qalqla': 'القلقلة',
            'safeer': 'الصفير',
            'itbaq': 'الإطباق',
            'tikraar': 'التكرار',
            'tafashie': 'التفشي',
            'istitala': 'الاستطالة'
        }
        
        ar_type_name = ar_types.get(error_type, error_type)
        
        return f"في كلمة '{word_text}': اختلاف في صفة {ar_type_name} - توقع '{t_expected}' لكن وُجد '{t_actual}'"
        
    def _extract_contextual_audio_clip(
        self,
        audio_path: str,
        word_index: int,
        word_timestamps: List[UserWordTimestamp],
    ) -> Optional[str]:
        """
        Extracts an audio clip including the word at word_index and its immediate neighbors
        using the user's own audio timestamps.
        """
        try:
            total_words = len(word_timestamps)
            if word_index < 0 or word_index >= total_words:
                return None

            # Determine the start and end words for the slice
            start_slice_index = word_index
            end_slice_index = word_index
            
            # Get the timestamps from the user's timing data
            start_ms = word_timestamps[start_slice_index].start_ms
            end_ms = word_timestamps[end_slice_index].end_ms

            # Load the audio and slice it
            full_audio = AudioSegment.from_file(audio_path)
            contextual_clip = full_audio[start_ms:end_ms]

            # Save the clip to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                contextual_clip.export(fp.name, format="mp3")
                return fp.name
        except Exception as e:
            self.logger.error(f"Failed to extract contextual audio for word index {word_index}: {e}")
            return None
    
    def get_correction_summary(self, result: CorrectionResult) -> str:
        """Generate a human-readable summary of the correction results."""
        if not result.errors_found:
            return f"ممتاز! لا توجد أخطاء في تلاوة الآية {result.ayah_number} من سورة رقم {result.surah_number}"
        
        summary = f"تم العثور على {len(result.errors_found)} خطأ في {result.total_words} كلمة:\n\n"
        
        for word_error in result.errors_found:
            summary += f"الكلمة: {word_error.word_text} (رقم {word_error.word_index + 1})\n"
            for error in word_error.errors:
                summary += f"  - {error.description}\n"
            summary += "\n"
        
        summary += f"دقة التلاوة الإجمالية: {result.overall_accuracy:.1%}"
        
        return summary


# Convenience functions
def correct_single_ayah(
    audio_path: Union[str, Path],
    surah_number: int,
    ayah_number: int,
    moshaf_attributes: Optional[MoshafAttributes] = None,
    device: str = "cpu"
) -> CorrectionResult:
    """
    Convenience function to correct a complete ayah.
    
    Args:
        audio_path: Path to student's audio recitation
        surah_number: Surah number (1-114)
        ayah_number: Ayah number
        moshaf_attributes: Optional moshaf configuration
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        CorrectionResult with detailed analysis
    """
    corrector = RecitationCorrector(moshaf_attributes=moshaf_attributes, device=device)
    return corrector.correct_recitation(audio_path, surah_number, ayah_number)


def correct_word_range(
    audio_path: Union[str, Path],
    surah_number: int,
    ayah_number: int,
    start_word_index: int,
    num_words: int,
    moshaf_attributes: Optional[MoshafAttributes] = None,
    device: str = "cpu"
) -> CorrectionResult:
    """
    Convenience function to correct a specific word range.
    
    Args:
        audio_path: Path to student's audio recitation
        surah_number: Surah number (1-114)
        ayah_number: Ayah number
        start_word_index: Starting word index (0-based)
        num_words: Number of words to analyze
        moshaf_attributes: Optional moshaf configuration
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        CorrectionResult with detailed analysis
    """
    corrector = RecitationCorrector(moshaf_attributes=moshaf_attributes, device=device)
    return corrector.correct_recitation(
        audio_path, surah_number, ayah_number, start_word_index, num_words
    )