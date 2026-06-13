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

def clean_template_artifacts(text):
    """
    Advanced text cleaner to strip out template artifacts completely.
    Removes Drucksache, Az., Abs., session references, faction names, dates,
    and simplifies template verbs/structures to look as natural as possible.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove speaker introductions
    text = re.sub(r'\bAls Abgeordneter\s+[A-ZÄÖÜa-zäöüß\-]+\s+[A-ZÄÖÜa-zäöüß\-]+\s+der Fraktion\s+[A-Z/a-zäöüß\-]+\s+(?:sehe ich mich in der Pflicht|betone ich|sehe ich)\b,?\s*(?:in dieser \d+\. Plenarsitzung)?\s*(?:beim Thema [^,]+)?\s* deutlich darauf hinzuweisen, dass\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAls Abgeordneter\s+der Fraktion\s+[A-Z/a-zäöüß\-]+\s+(?:sehe ich mich in der Pflicht|betone ich|sehe ich)\b,?\s*(?:in dieser \d+\. Plenarsitzung)?\s*(?:beim Thema [^,]+)?\s* deutliche darauf hinzuweisen, dass\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bIm Namen der Fraktion\s+[A-Z/a-zäöüß\-]+\s+betone ich\b,?\s*(?:in dieser \d+\. Plenarsitzung)?\s*(?:mit aller Deutlichkeit)?,?\s*dass\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAls Abgeordneter\s+[A-ZÄÖÜ\w\s\-]+ der Fraktion\s+\S+\b', '', text, flags=re.IGNORECASE)

    # 2. Remove Session References
    text = re.sub(r'\b(?:in der heutigen|in dieser|in der|heutigen)?\s*\d+\.\s*(?:plenarsitzung|sitzung)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:in dieser|in der|heutigen)?\s*plenarsitzung\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bplenarsitzung\b', '', text, flags=re.IGNORECASE)
    
    # 3. Remove Aktenzeichen (Az.)
    text = re.sub(r'\b(?:unter|gemäß|nach)?\s*Az\.\s*\d+/\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:unter|gemäß|nach)?\s*aktenzeichen\s*\d+/\d+\b', '', text, flags=re.IGNORECASE)
    
    # 4. Remove Paragraphs & Sections
    text = re.sub(r'\b(?:gemäß|nach|laut|nach)?\s*§+\s*\d+\s*(?:Abs\.|Absatz)\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:gemäß|nach|laut|nach)?\s*§+\s*\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:Abs\.|Absatz)\s*\d+\b', '', text, flags=re.IGNORECASE)
    
    # 5. Remove Printed Matter (Drucksache)
    text = re.sub(r'\b(?:in|gemäß|nach|laut)?\s*Drucksache\s*\d+(?:/\d+)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(?:in|gemäß|nach|laut)?\s*Drs\.\s*\d+(?:/\d+)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDrucksache\b', '', text, flags=re.IGNORECASE)
    
    # 6. Remove Dates
    text = re.sub(r'\b(?:am heutigen|heutigen)?\s*\d{2}\.\d{2}\.\d{4}\b', '', text)
    
    # 7. Clean up template phrases & keywords
    text = re.sub(r'\bbezüglich\b', 'für', text, flags=re.IGNORECASE)
    text = re.sub(r'\bund die Lasten einseitig abwälzt', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bund die Lasten einseitig abzuwälzen', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhochkrempeln\b', 'anpacken', text, flags=re.IGNORECASE)
    text = re.sub(r'\bzögert\b', 'wartet', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsachorientierte\b', 'konstruktive', text, flags=re.IGNORECASE)
    text = re.sub(r'\bvorbeigehen\b', 'vorübergehen', text, flags=re.IGNORECASE)
    text = re.sub(r'\bverspielt\b', 'verliert', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdrängenden\b', 'wichtigen', text, flags=re.IGNORECASE)
    text = re.sub(r'\bnachfolgende\b', 'kommende', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgemeinwohls\b', 'Allgemeinwohls', text, flags=re.IGNORECASE)
    text = re.sub(r'\babsichtserklärungen\b', 'Erklärungen', text, flags=re.IGNORECASE)
    text = re.sub(r'\bzielvorgaben\b', 'Zielen', text, flags=re.IGNORECASE)
    text = re.sub(r'\brichtungsentscheidung\b', 'Entscheidung', text, flags=re.IGNORECASE)
    
    # 8. Clean up extra punctuation/spaces resulting from removals
    text = re.sub(r'\s+', ' ', text)  # Collapse spaces
    text = re.sub(r'\s*,\s*,', ',', text)  # Collapse double commas
    text = re.sub(r',\s*\.', '.', text)  # Clean up comma-periods
    text = re.sub(r'^\s*,\s*', '', text)  # Remove leading commas
    text = re.sub(r'\b(?:in|beim Thema|zum Bereich)\s*[\.,]', '', text, flags=re.IGNORECASE)  # Remove orphaned prepositions
    
    # Clean up sentence starts
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
        
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
