
import sys
import os

# Adjust path to find src
sys.path.append(os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

try:
    from quran_transcript import quran_phonetizer, MoshafAttributes
except ImportError:
    print("Could not import quran_transcript. Checking paths.")
    sys.exit(1)

def inspect_ref():
    try:
        # Example text: "بِسْمِ"
        text = "بِسْمِ"
        # Dummy attributes
        attr = MoshafAttributes(
            rewaya="hafs",
            madd_monfasel_len=4,
            madd_mottasel_len=4,
            madd_mottasel_waqf=5,
            madd_aared_len=4
        ) 
        result = quran_phonetizer(text, attr, remove_spaces=True)
        
        print(f"Generated Sifat Length: {len(result.sifat)}")
        if result.sifat:
            first = result.sifat[0]
            print(f"Type: {type(first)}")
            print(f"Dir: {dir(first)}")
            print(f"Str: {str(first)}")
            
            # Check for specific attributes
            print(f"Has 'text'?: {hasattr(first, 'text')}")
            print(f"Has 'phonemes'?: {hasattr(first, 'phonemes')}")
            print(f"Has 'phonemes_group'?: {hasattr(first, 'phonemes_group')}")
            
            print(f"Value of 'text': {getattr(first, 'text', 'N/A')}")
            print(f"Value of 'phonemes': {getattr(first, 'phonemes', 'N/A')}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_ref()
