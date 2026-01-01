
import sys
import unittest
from dataclasses import dataclass
from typing import List, Any
from unittest.mock import MagicMock, patch
import re

sys.stdout.reconfigure(encoding='utf-8')

@dataclass
class Sifa:
    phonemes: str = ""
    phonemes_group: str = ""
    text: str = "" 
    
    def __str__(self):
        return f"Sifa(phonemes='{self.phonemes}', grp='{self.phonemes_group}')"

try:
    from src.quran_muaalem.correction_pipeline import RecitationCorrector
except ImportError:
    import os
    sys.path.append(os.path.abspath("."))
    from src.quran_muaalem.correction_pipeline import RecitationCorrector

class TestHarakaMismatch(unittest.TestCase):
    @patch('src.quran_muaalem.correction_pipeline.Muaalem')
    @patch('src.quran_muaalem.correction_pipeline.AlQuranAPIClient')
    @patch('src.quran_muaalem.correction_pipeline.TarteelAPIClient')
    @patch('src.quran_muaalem.correction_pipeline.AudioProcessor')
    def setUp(self, MockAudio, MockTarteel, MockAlQuran, MockMuaalem):
        self.corrector = RecitationCorrector(model_name_or_path="dummy", device="cpu")
        self.corrector.logger = MagicMock()
        
    def test_alhamdu_damma_vs_kasra_strict(self):
        # Scenario: Last letter of Alhamd(u).
        # Ref: Dal + Damma (u)
        # Pred: Dal + Kasra (i)
        
        # Test Case mirrors confirmed reality: 'phonemes_group' contains the vowel.
        ref_sifa = Sifa(phonemes="دُ", phonemes_group="دُ", text="دُ") 
        pred_sifa = Sifa(phonemes="دِ", phonemes_group="دِ", text="دِ")
        
        errors = self.corrector._analyze_word_errors(
            [pred_sifa], 
            [ref_sifa], 
            "دُ", 
            "Al-Hamd"
        )
        
        print("\nErrors Found:", len(errors))
        for e in errors:
            print(f"Type: {e.error_type}")
            print(f"Expect: {e.expected_value}")
            print(f"Actual: {e.actual_value}")
            print(f"Desc: {e.description}")
            
        self.assertTrue(len(errors) > 0, "Should detect error")
        self.assertTrue(any("دِ" in e.actual_value or "ِ" in e.actual_value for e in errors), "Actual value should show the error")

    def test_regex_cleaning(self):
        # Verify regex specifically
        txt = "دُ"
        # The fixer function: _clean_text_for_comparison
        # NOTE: Using a public mock or helper if available, or just testing logic
        cleaned = self.corrector._clean_text_for_comparison(txt, True)
        print(f"Cleaning 'دُ' -> '{cleaned}'")
        self.assertIn("ُ", cleaned, "Damma must be preserved!")

if __name__ == "__main__":
    unittest.main()
