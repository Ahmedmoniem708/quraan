import logging
import os
from dataclasses import asdict
import json
import tempfile
import requests
from pydub import AudioSegment
from typing import Literal, Optional, Any, get_origin, get_args
from pathlib import Path
import shutil

# Fix for LLVM SVML error on Windows - set environment variables early
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['PYTORCH_DISABLE_CUDA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
from quran_transcript.utils import PartOfUthmaniWord
from quran_transcript.phonetics.moshaf_attributes import (
    get_arabic_attributes,
    get_arabic_name,
)

# Import librosa with error handling
try:
    from librosa.core import load
    LIBROSA_AVAILABLE = True
except Exception as e:
    logging.warning(f"Librosa not available: {e}")
    LIBROSA_AVAILABLE = False
    def load(audio_path, sr=None, mono=True):
        raise RuntimeError("Librosa not available for audio loading")

from pydantic.fields import FieldInfo, PydanticUndefined
import gradio as gr

from .inference import Muaalem
from .muaalem_typing import MuaalemOutput
from .explain_gradio import explain_for_gradio
from .correction_pipeline import RecitationCorrector

# --- PATH SETUP ---
current_script_dir = Path(__file__).parent
project_root_dir = current_script_dir.parent.parent

# Audio Assets
INTRO_VOICE = project_root_dir / "assets" / "audio" / "intro_voice.mp3"
WRONG_VOICE = project_root_dir / "assets" / "audio" / "wrong_voice.mp3"
CORRECT_VOICE = project_root_dir / "assets" / "audio" / "correct_voice.mp3"

# JSON Data Paths
surah_json_path = project_root_dir / "surah.json"
segments_json_path = project_root_dir / "segments.json"

# Cache Directory
AUDIO_CACHE_DIR = ".audio_cache"
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# Load JSON data
try:
    with open(surah_json_path, 'r', encoding='utf-8') as f:
        SURAHS_DATA = json.load(f)
    with open(segments_json_path, 'r', encoding='utf-8') as f:
        SEGMENTS_DATA = json.load(f)
except FileNotFoundError as e:
    print(f"WARNING: JSON file not found at '{e.filename}'. Reference audio feature will not work.")
    SURAHS_DATA = {}
    SEGMENTS_DATA = {}

# Initialize components
REQUIRED_MOSHAF_FIELDS = [
    "rewaya", "takbeer", "madd_monfasel_len", "madd_mottasel_len",
    "madd_mottasel_waqf", "madd_aared_len", "madd_alleen_len",
    "ghonna_lam_and_raa", "meem_aal_imran", "madd_yaa_alayn_alharfy",
    "saken_before_hamz", "sakt_iwaja", "sakt_marqdena", "sakt_man_raq",
    "sakt_bal_ran", "sakt_maleeyah", "between_anfal_and_tawba",
    "noon_and_yaseen", "yaa_ataan", "start_with_ism", "yabsut",
    "bastah", "almusaytirun", "bimusaytir", "tasheel_or_madd",
    "yalhath_dhalik", "irkab_maana", "noon_tamnna", "harakat_daaf",
    "alif_salasila", "idgham_nakhluqkum", "raa_firq", "raa_alqitr",
    "raa_misr", "raa_nudhur", "raa_yasr", "meem_mokhfah",
]

model_id = "obadx/muaalem-model-v3_2"
logging.basicConfig(level=logging.INFO)
device = "cpu"
muaalem = Muaalem(model_name_or_path=model_id, device=device)
sampling_rate = 16000

# Initialize corrector
corrector = RecitationCorrector(device=device)

# Load Sura information
sura_idx_to_name = {}
sura_to_aya_count = {}
start_aya = Aya()
for sura_idx in range(1, 115):
    start_aya.set(sura_idx, 1)
    sura_idx_to_name[sura_idx] = start_aya.get().sura_name
    sura_to_aya_count[sura_idx] = start_aya.get().num_ayat_in_sura

# Default moshaf settings
default_moshaf = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=4,
)
current_moshaf = default_moshaf

# --- AUDIO PROCESSING FUNCTIONS ---

def merge_audio_sequence(user_context_path, ref_context_path):
    """
    Merges audio.
    If user_context_path exists: Intro -> User -> Wrong -> Correct -> Ref
    If user_context_path is None (Missing): Correct -> Ref
    """
    try:
        correct_phrase = AudioSegment.from_file(CORRECT_VOICE)
        ref_audio = AudioSegment.from_file(ref_context_path)
        pause = AudioSegment.silent(duration=500)

        if user_context_path:
            # Standard Case: User pronounced it wrong
            intro = AudioSegment.from_file(INTRO_VOICE)
            rong_phrase = AudioSegment.from_file(WRONG_VOICE)
            user_audio = AudioSegment.from_file(user_context_path)
            
            combined = intro + pause + user_audio + pause + rong_phrase + pause + correct_phrase + pause + ref_audio
        else:
            # Missing Word Case: Only play "The correct is..." + Ref
            combined = correct_phrase + pause + ref_audio
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            combined.export(fp.name, format="mp3")
            print(f"✅ تم إنشاء الفيديو المدمج في: {fp.name}")
            return fp.name
    except Exception as e:
        print(f"❌ خطأ في دمج الصوت: {e}")
        return None


def download_audio_if_needed(sura_number: int) -> str | None:
    """Downloads surah audio from a URL if not already in cache."""
    sura_key = str(sura_number)
    if sura_key not in SURAHS_DATA:
        print(f"Error: Surah {sura_number} not found in surah.json")
        return None

    audio_url = SURAHS_DATA[sura_key]["audio_url"]
    file_name = os.path.basename(audio_url)
    cached_file_path = os.path.join(AUDIO_CACHE_DIR, file_name)

    if not os.path.exists(cached_file_path):
        print(f"Downloading audio for Surah {sura_number}...")
        try:
            response = requests.get(audio_url)
            response.raise_for_status()
            with open(cached_file_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading audio for Surah {sura_number}: {e}")
            return None
    
    return cached_file_path


def download_audio_if_needed(sura_number: int) -> str | None:
    """Downloads surah audio from a URL if not already in cache. Returns the file path."""
    sura_key = str(sura_number)
    if sura_key not in SURAHS_DATA:
        print(f"Error: Surah {sura_number} not found in surah.json")
        return None

    audio_url = SURAHS_DATA[sura_key]["audio_url"]
    file_name = os.path.basename(audio_url)
    cached_file_path = os.path.join(AUDIO_CACHE_DIR, file_name)

    if not os.path.exists(cached_file_path):
        print(f"Downloading audio for Surah {sura_number}...")
        try:
            response = requests.get(audio_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(cached_file_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading audio for Surah {sura_number}: {e}")
            return None
    
    return cached_file_path


def get_reference_audio_clips(result) -> dict:
    """
    Downloads and slices reference audio for:
    1. The full verse.
    2. Isolated incorrect words.
    3. Contextual [previous, current, next] clips for incorrect words.
    """
    sura_num = result.surah_number
    ayah_num = result.ayah_number
    
    full_surah_path = download_audio_if_needed(sura_num)
    if not full_surah_path:
        return {}

    try:
        full_surah_audio = AudioSegment.from_mp3(full_surah_path)
    except Exception as e:
        print(f"Error loading audio file {full_surah_path} with pydub: {e}")
        return {}

    ayah_key = f"{sura_num}:{ayah_num}"
    if ayah_key not in SEGMENTS_DATA:
        print(f"Error: Ayah {ayah_key} not found in segments.json")
        return {}
    
    raw_segments = SEGMENTS_DATA[ayah_key]["segments"]
    
    # --- 3. BUILD ALIGNMENT MAP ---
    # We map the Text Index (Pipeline) -> Segment Index (Audio)
    # This handles cases where 2 text words = 1 audio segment.
    
    text_to_seg_map = {}
    
    # We get the list of words exactly as the pipeline sees them
    # result.ayah_text is the cleaned text (no Basmalah)
    words = result.ayah_text.split() 
    
    seg_ptr = 0
    text_ptr = 0
    
    while text_ptr < len(words):
        current_word = words[text_ptr]
        
        # Check if we are at a "Ya-Ayyuha" split
        # We check if the current word is the Vocative "Ya" and there is a next word
        is_ya_split = False
        if text_ptr + 1 < len(words):
            next_word = words[text_ptr + 1]
            
            # Check for the specific Uthmani drawing of "Ya" (يَـٰٓ or يَا)
            # and "Ayyuha" (أَيُّهَا)
            if ("يَـٰٓ" in current_word or "يَا" in current_word) and ("أَيُّهَا" in next_word):
                is_ya_split = True
        
        if is_ya_split:
            # MAP BOTH TEXT WORDS TO THE SAME SEGMENT
            text_to_seg_map[text_ptr] = seg_ptr      # Ya -> Segment X
            text_to_seg_map[text_ptr + 1] = seg_ptr  # Ayyuha -> Segment X
            
            # Advance text pointer by 2, but segment pointer by only 1
            text_ptr += 2
            seg_ptr += 1
        else:
            # Standard 1-to-1 Mapping
            text_to_seg_map[text_ptr] = seg_ptr
            text_ptr += 1
            seg_ptr += 1
            
    # ----------------------------------------
    
    # Slice the full verse
    verse_start_ms = raw_segments[0][1]
    verse_end_ms = raw_segments[-1][2]
    verse_audio_clip = full_surah_audio[verse_start_ms:verse_end_ms]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        verse_audio_clip.export(fp.name, format="mp3")
        verse_audio_path = fp.name

    # --- LOGIC FOR BOTH ISOLATED AND CONTEXTUAL CLIPS ---
    word_clips = []
    word_context_clips = []
    incorrect_word_indices = {error.word_index + 1 for error in result.errors_found}
    
    for error in result.errors_found:
        # Use the map to find the correct segment
        target_seg_idx = text_to_seg_map.get(error.word_index)
        
        if target_seg_idx is None:
            # Fallback for safety (though map should cover it)
            target_seg_idx = error.word_index
            
        # Boundary check
        if target_seg_idx >= len(raw_segments):
            print(f"⚠️ Segment index {target_seg_idx} out of range for Ayah {ayah_key}")
            continue
            
        try:
            # Get timestamps from the ALIGNED segment
            start_ms = raw_segments[target_seg_idx][1]
            end_ms = raw_segments[target_seg_idx][2]
            
            # Slice just this specific segment
            word_clip = full_surah_audio[start_ms:end_ms]
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                word_clip.export(fp.name, format="mp3")
                
                word_context_clips.append({
                    "word_index": error.word_index + 1, # Keep UI key consistent
                    "path": fp.name
                })
        except IndexError:
            print(f"⚠️ Error accessing segment data for word {error.word_index}")
            continue

    return {
        "word_context_clips": word_context_clips
    }


# --- GRADIO HELPER FUNCTIONS ---

def get_field_name(field_name: str, field_info: FieldInfo) -> str:
    label = field_name
    arabic_name = get_arabic_name(field_info)
    if arabic_name:
        label = f"{arabic_name} ({field_name})"
    return label


def create_gradio_input_for_field(field_name: str, field_info: FieldInfo, default_value: Any = None, help: str | None = None) -> Any:
    label = get_field_name(field_name, field_info)
    if default_value is None and field_info.default != PydanticUndefined:
        default_value = field_info.default
    if help is None:
        help = field_info.description

    if get_origin(field_info.annotation) is Literal:
        choices = list(get_args(field_info.annotation))
        arabic_attributes = get_arabic_attributes(field_info)
        choice_list = []
        for choice in choices:
            if arabic_attributes and choice in arabic_attributes:
                choice_list.append((arabic_attributes[choice], choice))
            else:
                choice_list.append((str(choice), choice))
        return gr.Dropdown(choices=choice_list, value=default_value, label=label, info=help, interactive=True)

    if field_info.annotation in [str, Optional[str]]:
        return gr.Textbox(value=default_value or "", label=label, info=help)
    elif field_info.annotation in [int, Optional[int]]:
        return gr.Number(value=default_value or 0, label=label, info=help, precision=0)
    elif field_info.annotation in [float, Optional[float]]:
        return gr.Number(value=default_value or 0.0, label=label, info=help, precision=1)
    elif field_info.annotation in [bool, Optional[bool]]:
        return gr.Checkbox(value=default_value or False, label=label, info=help)

    raise ValueError(f"Unsupported field type for {label}: {field_info.annotation}")


def update_aya_dropdown(sura_idx):
    if not sura_idx:
        sura_idx = 1
    return gr.update(choices=list(range(1, sura_to_aya_count[int(sura_idx)] + 1)), value=1)


def update_ayah_info(sura_idx, aya_idx):
    if not all([sura_idx, aya_idx]):
        return 1, ""
    try:
        sura_idx, aya_idx = int(sura_idx), int(aya_idx)
        aya_data = Aya(sura_idx, aya_idx).get()
        return len(aya_data.imlaey_words), aya_data.uthmani
    except Exception as e:
        return 1, f"Error loading Ayah data: {str(e)}"


def process_audio(audio, sura_idx, aya_idx, start_idx, num_words):
    global current_moshaf
    if audio is None:
        return "Please upload an audio file first"
    try:
        print("\n" + "="*40)
        print("🔍 CLASSIC ANALYSIS DEBUGGER")
        print("="*40)
        
        # 1. Prepare Text
        uthmani_ref = (Aya(int(sura_idx), int(aya_idx)).get_by_imlaey_words(int(start_idx), int(num_words)).uthmani)
        print(f"1️⃣  Uthmani Text: {uthmani_ref}")
        
        # 2. Phoneticize (Generate Reference)
        phonetizer_out = quran_phonetizer(uthmani_ref, current_moshaf, remove_spaces=True)
        print(f"2️⃣  Reference Phonemes (Input to Model):")
        print(f"   {phonetizer_out.phonemes}")
        
        # 3. Load Audio
        wave, _ = load(audio, sr=sampling_rate, mono=True)
        print(f"3️⃣  Audio Loaded: {len(wave)} samples (Duration: {len(wave)/sampling_rate:.2f}s)")
        
        # 4. Run Model
        print("4️⃣  Running Model Inference...")
        outs = muaalem([wave], [phonetizer_out], sampling_rate=sampling_rate)
        
        # 5. Inspect Output
        model_output = outs[0]
        print("\n5️⃣  MODEL OUTPUT RECEIVED:")
        print(f"   Detected Phonemes Text: {model_output.phonemes.text}")
        print(f"   Total Phoneme Groups: {len(model_output.sifat)}")
        
        if len(model_output.sifat) > 0:
            print("\n   --- Example Data (First Phoneme) ---")
            first_sifa = model_output.sifat[0]
            print(f"   Phoneme Group: {getattr(first_sifa, 'phonemes_group', 'N/A')}")
            print(f"   Hams/Jahr: {first_sifa.hams_or_jahr}")
            print(f"   Qalqala: {first_sifa.qalqla}")
            print("   ------------------------------------")

        print("="*40 + "\n")
        
        # Add explanation        
        return explain_for_gradio(outs[0].phonemes.text, phonetizer_out.phonemes, outs[0].sifat, phonetizer_out.sifat, lang="arabic")
    except PartOfUthmaniWord as e:
        return f"⚠️ Error: Partial Uthmani words selected. {str(e)}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing audio: {str(e)}"


def generate_attributes_card(word_error) -> str:
    """
    Generates a visually appealing HTML card with full Arabic translation 
    for attributes, values, and phonetic symbols.
    """
    if not word_error or not word_error.errors:
        return ""

    # --- 1. TRANSLATION DICTIONARIES ---
    
    # Map internal variable names to Arabic UI labels
    ATTR_MAP = {
        'articulation': 'خطأ في الحرف (المخرج)',  # <--- NEW ADDITION
        'hams_or_jahr': 'الهمس والجهر',
        'shidda_or_rakhawa': 'الشدة والرخاوة',
        'tafkheem_or_taqeeq': 'التفخيم والترقيق',
        'itbaq': 'الإطباق',
        'safeer': 'الصفير',
        'qalqla': 'القلقلة',
        'tikraar': 'التكرار',
        'tafashie': 'التفشي',
        'istitala': 'الاستطالة',
        'ghonna': 'الغنة',
    }

    # Map values to Arabic
    VAL_MAP = {
        # Values
        'hams': 'همس',
        'jahr': 'جهر',
        'shadeed': 'شديد',
        'rikhw': 'رخو',
        'between': 'توسط (بينية)',
        'mufakham': 'مفخم',
        'muraqaq': 'مرقق',
        'motbaq': 'مطبق',
        'monfateh': 'منفتح',
        'safeer': 'صفير',
        'no_safeer': 'لا يوجد صفير',
        'moqalqal': 'مقلقل',
        'not_moqalqal': 'غير مقلقل',
        'mokarar': 'مكرر',
        'not_mokarar': 'غير مكرر',
        'motafashie': 'متفشي',
        'not_motafashie': 'غير متفشي',
        'mostateel': 'مستطيل',
        'not_mostateel': 'غير مستطيل',
        'maghnoon': 'غنة ظاهرة',
        'not_maghnoon': 'لا توجد غنة',
        # Fallbacks
        'None': 'لا يوجد',
        'nan': 'غير محدد'
    }

    # Map Symbols (Diacritics) to Names
    PHONEME_MAP = {
        'َ': 'فتحة',
        'ِ': 'كسرة',
        'ُ': 'ضمة',
        'ْ': 'سكون',
        'ّ': 'شدة',
        'ٰ': 'ألف خنجربة',
        'a': 'فتحة',
        'u': 'ضمة',
        'i': 'كسرة',
        '~': 'مد',
    }

    # --- 2. HTML GENERATION ---

    html = f"""
    <div style="direction: rtl; text-align: right; font-family: 'Amiri', 'Segoe UI', serif; 
                margin-top: 8px; margin-bottom: 15px; 
                padding: 10px 15px; 
                background: linear-gradient(to left, #fffbfb, #ffffff); 
                border-radius: 8px; 
                border-right: 4px solid #e53e3e; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        
        <div style="display: flex; align-items: center; margin-bottom: 10px; border-bottom: 1px solid #eee; padding-bottom: 5px;">
            <span style="font-size: 1.2em; font-weight: bold; color: #742a2a;">
                📋 تحليل الأخطاء في كلمة: <span style="color: #c53030; font-family: 'Traditional Arabic'; font-size: 1.3em;">"{word_error.word_text}"</span>
            </span>
        </div>
        
        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
    """

    for error in word_error.errors:
        # 1. Clean up the Attribute Name
        attr_display = ATTR_MAP.get(error.error_type, error.error_type)
        
        # 2. Clean up Expected/Actual Values
        expected_display = VAL_MAP.get(str(error.expected_value), str(error.expected_value))
        actual_display = VAL_MAP.get(str(error.actual_value), str(error.actual_value))
        
        # 3. Handle Phoneme/Letter Display
        raw_phoneme = error.phoneme_group
        if raw_phoneme and raw_phoneme != "None":
            # Check if it's a symbol/diacritic and get its name
            if raw_phoneme in PHONEME_MAP:
                phoneme_display = f"حركة: {PHONEME_MAP[raw_phoneme]}"
            else:
                phoneme_display = f"الحرف: {raw_phoneme}"
                
            letter_html = f"""
            <div style="margin-bottom: 6px; border-bottom: 1px dashed #cbd5e0; padding-bottom: 4px;">
                <span style="background: #edf2f7; color: #2d3748; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 1.0em;">
                    {phoneme_display}
                </span>
            </div>
            """
        else:
            letter_html = ""

        html += f"""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 6px; 
                    padding: 10px; min-width: 180px; flex: 1;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
            
            {letter_html}
            
            <div style="color: #c53030; font-weight: bold; font-size: 1em; margin-bottom: 6px;">
                ⚠️ {attr_display}
            </div>
            
            <div style="font-size: 0.9em; line-height: 1.6;">
                <div style="background: #f0fff4; color: #1F1F1F; padding: 2px 5px; border-radius: 4px; margin-bottom: 2px;">
                    <span style="font-weight: bold; color: #276749;">✅ الصواب:</span> {expected_display}
                </div>
                <div style="background: #fff5f5; color: #1F1F1F; padding: 2px 5px; border-radius: 4px;">
                    <span style="font-weight: bold; color: #c53030;">❌ خطؤك:</span> {actual_display}
                </div>
                <div style="color: #718096; font-size: 0.8em; margin-top: 5px; text-align: left;">
                    الثقة: {error.confidence:.0%}
                </div>
            </div>
        </div>
        """

    html += """
        </div>
    </div>
    """
    return html


# --- MAIN CORRECTION LOGIC ---

def process_audio_with_correction(audio, sura_idx, aya_idx, start_idx, num_words):
    """
    Handles only the essential outputs:
    1. HTML Analysis
    2. Contextual Players (User, Ref, Merged)
    3. Comparison HTML
    """
    global corrector
    MAX_CONTEXT_PLAYERS = 300

    outputs = {
        "analysis_html": "Error: Please upload an audio file.",
        "comparison_html": ""
    }
    # Initialize all to hidden
    for i in range(MAX_CONTEXT_PLAYERS):
        outputs[f"user_p_{i}"] = gr.update(visible=False)
        outputs[f"missing_msg_{i}"] = gr.update(visible=False) # NEW
        outputs[f"ref_p_{i}"] = gr.update(visible=False)
        outputs[f"merged_p_{i}"] = gr.update(visible=False)
        # NEW: Initialize Attribute HTML
        outputs[f"attr_html_{i}"] = gr.update(visible=False, value="")
    
    def format_outputs():
        return (
            outputs["analysis_html"],
            *[outputs.get(f"user_p_{i}", gr.update(visible=False)) for i in range(MAX_CONTEXT_PLAYERS)],
            *[outputs.get(f"missing_msg_{i}", gr.update(visible=False)) for i in range(MAX_CONTEXT_PLAYERS)], # NEW
            *[outputs.get(f"ref_p_{i}", gr.update(visible=False)) for i in range(MAX_CONTEXT_PLAYERS)],
            *[outputs.get(f"merged_p_{i}", gr.update(visible=False)) for i in range(MAX_CONTEXT_PLAYERS)],
            # NEW: Return Attribute HTMLs
            *[outputs.get(f"attr_html_{i}", gr.update(visible=False)) for i in range(MAX_CONTEXT_PLAYERS)],
            outputs["comparison_html"]
        )

    if audio is None:
        outputs["analysis_html"] = "Error: Please upload an audio file."
        return format_outputs()

    try:
        result = corrector.correct_recitation(
            audio_path=audio, surah_number=int(sura_idx), ayah_number=int(aya_idx),
            start_word_index=int(start_idx), num_words=int(num_words) if num_words else None
        )

        outputs["analysis_html"] = generate_correction_html(result)

        if result.errors_found:
            reference_clips = get_reference_audio_clips(result)
            context_ref_map = {clip['word_index']: clip['path'] for clip in reference_clips.get("word_context_clips", [])}

            for idx, error in enumerate(result.errors_found):
                if idx >= MAX_CONTEXT_PLAYERS: break

                user_context_path = getattr(error, "user_context_audio_path", None)
                ref_context_path = context_ref_map.get(error.word_index + 1)

                # PLAYER 1: User Context OR Missing Message
                if user_context_path:
                    # User pronounced the word
                    outputs[f"user_p_{idx}"] = gr.update(
                        visible=True, value=user_context_path, label=f"تلاوتك: {error.word_text}"
                    )
                    outputs[f"missing_msg_{idx}"] = gr.update(visible=False)
                else:
                    # Word Missing
                    outputs[f"user_p_{idx}"] = gr.update(visible=False)
                    outputs[f"missing_msg_{idx}"] = gr.update(
                        visible=True, 
                        value=f"### ⚠️ لم تقم بنطق كلمة: {error.word_text}"
                    )

                # PLAYER 2: Reference Context (Always show if available)
                if ref_context_path:
                    outputs[f"ref_p_{idx}"] = gr.update(
                        visible=True, value=ref_context_path, label=f"الشيخ الحصري: {error.word_text}"
                    )
                
                # PLAYER 3: Merged Audio
                # Logic handles both cases (User exists or None)
                if ref_context_path:
                    merged_path = merge_audio_sequence(user_context_path, ref_context_path)
                    if merged_path:
                        outputs[f"merged_p_{idx}"] = gr.update(
                            visible=True, value=merged_path, label=f"المعلم (مدمج): {error.word_text}"
                        )
                        
                        # 4. NEW: Generate Attributes HTML Card
                        # Only show if there are specific attribute errors (not just missing word)
                        # If word is missing, attributes might be empty or 'missing_word' type.
                        # You can choose to show it or not. The generator handles logic.
                        attr_card_html = generate_attributes_card(error)
                        outputs[f"attr_html_{idx}"] = gr.update(visible=True, value=attr_card_html)

        return format_outputs()

    except Exception as e:
        import traceback
        traceback.print_exc()
        outputs["analysis_html"] = f"❌ Error in application: {str(e)}"
        return format_outputs()


def generate_correction_html(result):
    """Generate clean HTML report."""
    html = f"""
    <div style="background: #7F7F7F; direction: rtl; text-align: right; font-family: 'Amiri', 'Traditional Arabic', serif;">
        <h3>🎯 نتائج فحص التلاوة</h3>
        
        <div style="background: #7F7F7F; color: black; padding: 15px; border-radius: 10px; margin: 10px 0;">
            <h4>📖 معلومات الآية</h4>
            <p><strong>السورة:</strong> {result.surah_number} | <strong>الآية:</strong> {result.ayah_number}</p>
            <p><strong>النص:</strong> {result.ayah_text}</p>
            <p><strong>دقة التلاوة:</strong> {result.overall_accuracy:.1%}</p>
        </div>
    """
    
    if result.errors_found:
        html += """
        <div style="background: #C03C35; color: black; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #e53e3e;">
            <h4 style="color: #c53030;">⚠️ الأخطاء المكتشفة</h4>
        """
        
        for i, word_error in enumerate(result.errors_found, 1):
            html += f"""
            <div style="background: #7F7F7F; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #fed7d7;">
                <h5>خطأ {i}: الكلمة "{word_error.word_text}" (موضع {word_error.word_index})</h5>
                <ul>
            """
            for error in word_error.errors:
                html += f"""
                <li style="margin: 5px 0;">
                    <strong>{error.error_type}:</strong> {error.description}
                    <br><small>المتوقع: {error.expected_value} | الفعلي: {error.actual_value}</small>
                </li>
                """
            html += "</ul></div>"
        html += "</div>"
    else:
        html += """
        <div style="background: #7F7F7F; color: black; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #38a169;">
            <h4 style="color: #2f855a;">✅ ممتاز!</h4>
            <p>لم يتم اكتشاف أي أخطاء في التلاوة.</p>
        </div>
        """
    
    html += f"""
    <div style="background: #7F7F7F; color: black; padding: 15px; border-radius: 10px; margin: 10px 0;">
        <h4>📊 الملخص</h4>
        <p>{corrector.get_correction_summary(result)}</p>
    </div>
    </div>
    """
    return html


def update_moshaf_settings(*args):
    global current_moshaf, field_names
    try:
        settings_dict = dict(zip(field_names, args))
        current_moshaf = MoshafAttributes(**settings_dict)
        return "✅ تم حفظ الإعدادات بنجاح!"
    except Exception as e:
        return f"❌ خطأ في حفظ الإعدادات: {str(e)}"


def reset_settings():
    global current_moshaf
    try:
        current_moshaf = default_moshaf
        default_values = [getattr(default_moshaf, field_name) for field_name in field_names]
        return default_values + ["✅ تم إعادة التعيين!"]
    except Exception as e:
        return [getattr(current_moshaf, fn) for fn in field_names] + [f"❌ Error: {str(e)}"]


# --- GRADIO UI LAYOUT ---

with gr.Blocks(title="المعلم القرآني") as app:
    current_moshaf_state = gr.State(default_moshaf)
    field_names = []

    with gr.Tab("التحليل الرئيسي"):
        gr.Markdown("# كشف أخطاء التلاوة والتجويد")

        with gr.Row():
            with gr.Column(scale=1):
                sura_choices = [(f"{idx} - {sura_idx_to_name[idx]}", idx) for idx in range(1, 115)]
                sura_dropdown = gr.Dropdown(choices=sura_choices, label="السورة", value=1)
                aya_dropdown = gr.Dropdown(choices=list(range(1, sura_to_aya_count[1] + 1)), label="رقم الآية", value=1)
                
                # Hidden/Status inputs
                start_idx = gr.Number(value=0, visible=False)
                num_words = gr.Number(value=1, label="عدد كلمات الآية", interactive=False)
                uthmani_text = gr.Textbox(label="الرسم العثماني", interactive=False)

            with gr.Column(scale=2):
                audio_input = gr.Audio(sources=["upload", "microphone"], label="تسجيل التلاوة", type="filepath")
                
                with gr.Row():
                    analyze_btn = gr.Button("افحص التلاوة (تحليل متقدم)", variant="primary")
                    classic_analyze_btn = gr.Button("التحليل الكلاسيكي", variant="secondary")
                
                output_html = gr.HTML(label="نتيجة الفحص")

                # --- ONLY CONTEXT PLAYERS & MERGED PLAYERS REMAIN ---
                MAX_CONTEXT_PLAYERS = 300
                user_context_players = []
                missing_msgs = []
                ref_context_players = []
                merged_context_players = []
                # NEW: Attribute HTML List
                attr_htmls = [] 

                for i in range(MAX_CONTEXT_PLAYERS):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # User Player OR Text Message
                            user_p = gr.Audio(label=f"تلاوتك {i+1}", visible=False)
                            msg = gr.Markdown(visible=False) # Text replacement
                        with gr.Column(scale=1):
                            ref_p = gr.Audio(label=f"المرجع {i+1}", visible=False)
                    
                    merged_p = gr.Audio(label=f"المعلم المدمج {i+1}", visible=False)
                    
                    # NEW: Add the HTML component below the merged audio
                    attr_h = gr.HTML(visible=False) 
                    
                    gr.HTML("<hr>") # Separator

                    user_context_players.append(user_p)
                    missing_msgs.append(msg)
                    ref_context_players.append(ref_p)
                    merged_context_players.append(merged_p)
                    attr_htmls.append(attr_h)
                
                comparison_html_output = gr.HTML(visible=False) # Placeholder if needed later

        # Wiring inputs/outputs
        ayah_info_inputs = [sura_dropdown, aya_dropdown]
        ayah_info_outputs = [num_words, uthmani_text]

        app.load(fn=update_ayah_info, inputs=ayah_info_inputs, outputs=ayah_info_outputs)
        sura_dropdown.change(fn=update_aya_dropdown, inputs=sura_dropdown, outputs=aya_dropdown)\
                     .then(fn=update_ayah_info, inputs=ayah_info_inputs, outputs=ayah_info_outputs)
        aya_dropdown.change(fn=update_ayah_info, inputs=ayah_info_inputs, outputs=ayah_info_outputs)

        analyze_btn.click(
            fn=process_audio_with_correction,
            inputs=[audio_input, sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=[
                output_html,
                *user_context_players,
                *missing_msgs,
                *ref_context_players,
                *merged_context_players,
                *attr_htmls, # <--- NEW: Add to outputs
                comparison_html_output
            ],
        )
        
        classic_analyze_btn.click(
            process_audio,
            inputs=[audio_input, sura_dropdown, aya_dropdown, start_idx, num_words],
            outputs=output_html,
        )

    with gr.Tab("إعدادات المصحف"):
        settings_components = []
        fields = MoshafAttributes.model_fields
        for field_name in REQUIRED_MOSHAF_FIELDS:
            field_info = fields[field_name]
            input_component = create_gradio_input_for_field(field_name, field_info, getattr(default_moshaf, field_name, None))
            settings_components.append(input_component)
            field_names.append(field_name)

        with gr.Row():
            save_btn = gr.Button("حفظ الإعدادات", variant="primary")
            reset_btn = gr.Button("إعادة التعيين")
        status_message = gr.Markdown()

        save_btn.click(update_moshaf_settings, inputs=settings_components, outputs=status_message)
        reset_btn.click(reset_settings, inputs=[], outputs=settings_components + [status_message])

def main():
    app.launch(server_name="127.0.0.1", share=True)

if __name__ == "__main__":
    main()