
import sys
import unittest
from dataclasses import dataclass, field
from typing import List, Any
from unittest.mock import MagicMock, patch
import logging

sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class Sifa:
    phonemes_group: str = ""
    text: str = "" 
    def __str__(self): return self.phonemes_group

@dataclass
class UserWordTimestamp:
    word: str
    start_ms: int
    end_ms: int

# Mock classes to allow import
sys.modules['src.quran_muaalem.inference'] = MagicMock()
sys.modules['src.quran_muaalem.muaalem_typing'] = MagicMock()
sys.modules['src.quran_muaalem.api_clients'] = MagicMock()
sys.modules['src.quran_muaalem.audio_processing'] = MagicMock()

# Mock quran_transcript and its submodules
qt_mock = MagicMock()
alphabet_mock = MagicMock()
utils_mock = MagicMock()
sys.modules['quran_transcript'] = qt_mock
sys.modules['quran_transcript.alphabet'] = alphabet_mock
sys.modules['quran_transcript.utils'] = utils_mock

try:
    # We need to hack sys.path if running from separate dir, but usually '.' is fine
    # Import the actual class under test
    from src.quran_muaalem.correction_pipeline import RecitationCorrector
except ImportError:
    import os
    sys.path.append(os.path.abspath("."))
    from src.quran_muaalem.correction_pipeline import RecitationCorrector

class TestSkippedWord(unittest.TestCase):
    @patch('src.quran_muaalem.correction_pipeline.Muaalem')
    @patch('src.quran_muaalem.correction_pipeline.AlQuranAPIClient')
    @patch('src.quran_muaalem.correction_pipeline.TarteelAPIClient')
    @patch('src.quran_muaalem.correction_pipeline.AudioProcessor')
    @patch('src.quran_muaalem.correction_pipeline.quran_phonetizer')
    def setUp(self, MockPhonetizer, MockAudio, MockTarteel, MockAlQuran, MockMuaalem):
        self.corrector = RecitationCorrector(model_name_or_path="dummy", device="cpu")
        self.corrector.logger = MagicMock()
        
        # Mock Phonetizer to return length info
        def side_effect(text, *args, **kwargs):
            m = MagicMock()
            m.phonemes = text # Use text as phonemes for simplicity
            return m
        MockPhonetizer.side_effect = side_effect

    def test_skipped_word_alignment(self):
        # Scenario: 
        # Ref Words: [A, B, C]
        # User Words: [A, C] (Skipped B)
        # Expected: Mapping A->TS_A, B->None, C->TS_C
        
        ref_words = ["Alhamdu", "Lillahi", "Rabb"]
        user_ts = [
            UserWordTimestamp(word="Alhamdu", start_ms=0, end_ms=1000),
            UserWordTimestamp(word="Rabb", start_ms=2000, end_ms=3000)
        ]
        
        # Test 1: Verify Alignment Logic
        alignment_map = self.corrector._align_detected_words(user_ts, ref_words)
        print("\nAlignment Map Keys:", alignment_map.keys())
        print("Alignment Map Values (Words):", [getattr(t, 'word', 'None') if t else 'None' for t in alignment_map.values()])
        
        self.assertIsNotNone(alignment_map.get(0), "Word 0 (Alhamdu) should match")
        self.assertIsNone(alignment_map.get(1), "Word 1 (Lillahi) should be None (Skipped)")
        self.assertIsNotNone(alignment_map.get(2), "Word 2 (Rabb) should match")
        self.assertEqual(alignment_map[2].word, "Rabb")

    def test_timestamp_mapping_logic(self):
        # Test _map_phonemes_to_words_by_parts with aligned timestamps
        ref_words = ["Alhamdu", "Lillahi", "Rabb"]
        # Aligned TS list: [TS_A, None, TS_C]
        ts_a = UserWordTimestamp(word="Alhamdu", start_ms=0, end_ms=1000)
        ts_c = UserWordTimestamp(word="Rabb", start_ms=2000, end_ms=3000)
        aligned_ts = [ts_a, None, ts_c]
        
        # Predicted Sifat (Mock)
        # Duration 3s. Total 30 items.
        # 0-1s (0-10): Matches A
        # 1-2s (10-20): Gap/Missing (matches B roughly in time but TS says None)
        # 2-3s (20-30): Matches C
        pred_sifat = [Sifa(f"s{i}") for i in range(30)]
        
        # Full Ref Phonetic Mock
        full_ref = MagicMock()
        full_ref.sifat = [Sifa(f"r{i}") for i in range(15)] # partial match
        full_ref.phonemes = "ABC"

        res = self.corrector._map_phonemes_to_words_by_parts(
            pred_sifat,
            ref_words,
            aligned_ts,
            3.0, # seconds duration
            full_ref
        )
        
        word_to_pred_parts, word_to_pred_sifat, _, word_to_ts_idx, _ = res
        
        # Word 0
        self.assertEqual(word_to_ts_idx[0], (0, 10))
        self.assertEqual(len(word_to_pred_sifat[0]), 10)
        
        # Word 1 (Skipped)
        self.assertEqual(word_to_ts_idx[1], (0, 0))
        self.assertEqual(len(word_to_pred_sifat[1]), 0)
        
        # Word 2
        self.assertEqual(word_to_ts_idx[2], (20, 30))
        self.assertEqual(len(word_to_pred_sifat[2]), 10)
        
        print("\nMapping Logic Verified: Skipped word mapped to Empty Sifat.")

if __name__ == "__main__":
    unittest.main()
