import json
import csv
import re
import os

# Custom stop words representing template leak keywords in lower case
TEMPLATE_KEYWORDS = {
    'plenarsitzung', 'hochkrempeln', 'zögert', 'bezüglich', 'abwälzt', 
    'sachorientierte', 'vorbeigehen', 'verspielt', 'drängenden', 'nachfolgende', 
    'gemeinwohls', 'absichtserklärungen', 'zielvorgaben', 'richtungsentscheidung', 
    'überarbeitung', 'lebensrealität', 'hintertreffen', 'entwurfs', 'kulturförderung',
    'ärmel', 'bsw'
}

def clean_text(text):
    """
    Applies regex masking to replace dates, session terms, party names, and numbers 
    with generic placeholder tokens. This forces models to learn grammatical and 
    stylistic writing patterns rather than specific template-specific keywords.
    """
    if not text:
        return ""
    
    # 1. Mask Dates (e.g. 09.10.2025 or 15.08.2019)
    text = re.sub(r'\b\d{2}\.\d{2}\.\d{4}\b', '[DATUM]', text)
    
    # 2. Mask Bundestag session references (e.g. 137. Plenarsitzung or Plenarsitzung)
    text = re.sub(r'\b\d+\.\s*(?:plenarsitzung|sitzung)\b', '[SITZUNG]', text, flags=re.IGNORECASE)
    text = re.sub(r'\bplenarsitzung\b', '[SITZUNG]', text, flags=re.IGNORECASE)
    
    # 3. Mask Faction/Party names
    # Includes CDU/CSU, SPD, AfD, FDP, Grüne, BSW, Linke
    text = re.sub(r'\b(?:cdu/csu|cdu|csu|spd|afd|fdp|grüne|bsw|linke)\b', '[PARTEI]', text, flags=re.IGNORECASE)
    
    # 4. Mask remaining numbers
    text = re.sub(r'\b\d+\b', '[ZAHL]', text)
    
    return text

def load_ai_data_generator(filepath):
    """
    Yields data records from the alternating-line AI JSONL dataset.
    Odd lines (1-indexed) contain the text data.
    Even lines contain generation metadata (top_p, seed, etc.) which are skipped.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"AI dataset not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx % 2 == 0:  # Data line (0-indexed 0, 2, 4...)
                try:
                    record = json.loads(line.strip())
                    yield {
                        'text': record.get('text', ''),
                        'label': int(record.get('label', 1)),
                        'domain': record.get('style', 'unknown'),  # style maps to domain
                        'source_type': record.get('style', 'unknown'),
                        'document_id': f"ai_{idx//2}",
                        'speaker': record.get('model', 'ai_model'),
                        'url': record.get('provider', 'ai_provider')
                    }
                except Exception as e:
                    pass

def load_human_data_generator(filepath):
    """
    Yields data records from the Human CSV dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Human dataset not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            yield {
                'text': row.get('text', ''),
                'label': int(row.get('label', 0)),
                'domain': row.get('domain', ''),
                'source_type': row.get('source_type', ''),
                'document_id': row.get('document_id', ''),
                'speaker': row.get('speaker', ''),
                'url': row.get('url', '')
            }
