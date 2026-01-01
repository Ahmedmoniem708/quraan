import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

text = "دُ" # Dal + Damma
cleaned = re.sub(r'[^\w]', '', text)
print(f"Original: {text}")
print(f"Cleaned: {cleaned}")
print(f"Kept Damma? {'Yes' if 'ُ' in cleaned else 'No'}")
