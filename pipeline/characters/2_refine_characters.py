import json
import re
import os
import torch
from transformers import pipeline
from tqdm import tqdm

# On charge un modèle NER (Named Entity Recognition)
# 'dbmdz/bert-large-cased-finetuned-conll03-english' est excellent pour les noms propres
print("🚀 Chargement du modèle de reconnaissance d'entités (NER)...")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)

METADATA_KEYS = [
    'weapons of choice', 'weapon of choice', 'combat style', 'cursed technique', 
    'devil fruit type', 'devil fruit', 'devil contracts', 'blood type', 
    'teams', 'team', 'height', 'year', 'class', 'number', 'position', 'occupation', 
    'affiliation', 'affiliations', 'relatives', 'relative', 'family', 'guild', 'guilds', 'bounty', 'age', 'born', 'species', 'kind', 
    'nationality', 'birthday', 'gender', 'status', 'grade', 'alias', 
    'ability', 'breathing', 'weapon', 'weapons', 'race', 'zanpakutou', 'debut', 'residence', 'hardness', 'series', 'title', 'powers', 'rank', 'partner', 'franxx', 'quirk', 'birthplace'
]

def clean_markdown_links(text):
    return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

def get_popularity_rank(favs):
    if favs >= 10000: return 5
    if favs >= 2000: return 4
    if favs >= 500: return 3
    if favs >= 100: return 2
    return 1

def smart_split(text):
    parts = []
    current = []
    paren_depth = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == '(': paren_depth += 1
        elif char == ')': paren_depth -= 1
        if paren_depth == 0:
            if text[i:i+5].lower() == ' and ':
                parts.append("".join(current).strip()); current = []; i += 5; continue
            if char in [',', ';']:
                parts.append("".join(current).strip()); current = []; i += 1; continue
        current.append(char)
        i += 1
    parts.append("".join(current).strip())
    return [p for p in parts if len(p) > 1]

def extract_metadata_v89(text, char_name):
    metadata = {}
    name_parts = char_name.split()
    first_name = name_parts[0]
    
    # Header Extraction (Regex Multi-Pass)
    meta_block_pattern = r'(?:__|(?:\*\*))([^:_]+):?(?:__|(?:\*\*))\s*(.*?)(?=(?:__|(?:\*\*))|\n|$)'
    matches = list(re.finditer(meta_block_pattern, text))
    last_meta_idx = 0
    for m in matches:
        k_raw, v_raw = m.groups()
        if k_raw.lower() in METADATA_KEYS:
            # Rupture narrative
            stop_m = re.search(rf'\b(?:{re.escape(char_name)}|{re.escape(first_name)})\b|' + r'\b(?:He|She|They|It|Born|Initially)\b\s+(?:is|was|has|in)', v_raw, re.I)
            v_ext = v_raw[:stop_m.start()] if stop_m else v_raw
            
            t_key = 'affiliations' if any(x in k_raw.lower() for x in ['team', 'affiliation', 'club', 'guild']) else ('occupation' if any(x in k_raw.lower() for x in ['occupation', 'position']) else ('powers' if any(x in k_raw.lower() for x in ['power', 'weapon', 'technique', 'fruit', 'ability', 'breath']) else k_raw.lower().replace(' ', '_')))
            
            if t_key in ['affiliations', 'occupation', 'relatives', 'powers', 'kind']:
                parts = smart_split(v_ext)
                if parts: metadata[t_key] = list(set(metadata.get(t_key, []) + parts))
            else:
                metadata[t_key] = v_ext.strip()
            
            last_meta_idx = max(last_meta_idx, m.start() + m.group(0).find(v_ext) + len(v_ext))
            if stop_m: break

    remaining = text[last_meta_idx:].strip()
    first_major = re.search(rf'\b(?:{re.escape(char_name)}|{re.escape(first_name)})\b|[A-Z]', remaining)
    biography = remaining[first_major.start():] if first_major else remaining
    return metadata, biography

def extract_entities_ia(biography, char_name):
    """Utilise l'IA pour extraire les lieux, organisations et personnages cités."""
    if not biography or len(biography) < 20:
        return [], [], []

    # Le modèle NER peut être lent, on limite à la première partie de la bio si elle est géante
    input_text = biography[:512] 
    entities = ner_pipeline(input_text)
    
    related_people = []
    locations = []
    organizations = []
    
    name_parts = char_name.lower().split()

    for ent in entities:
        label = ent['entity_group']
        word = ent['word'].strip()
        
        # Filtrage : On évite de s'extraire soi-même
        if any(np in word.lower() for np in name_parts) or word.lower() in name_parts:
            continue

        if label == 'PER': # Personne
            if len(word) > 2: related_people.append(word)
        elif label == 'LOC' or label == 'GPE': # Lieu / Pays
            if len(word) > 2: locations.append(word)
        elif label == 'ORG': # Organisation / Clan
            if len(word) > 2: organizations.append(word)
            
    return list(set(related_people)), list(set(locations)), list(set(organizations))

def refine_character(char):
    raw_desc = char.get('description', '')
    raw_desc = raw_desc.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\ufeff', '')
    name = char.get('title', 'Unknown')
    
    text_no_md = clean_markdown_links(raw_desc)
    text_no_md = re.sub(r'~!|!~', '', text_no_md)
    
    metadata, biography = extract_metadata_v89(text_no_md, name)
    
    # EXTRACTION IA
    peop, locs, orgs = extract_entities_ia(biography, name)
    
    # Fusion avec les related_characters extraits via Markdown précédemment
    md_related = re.findall(r'\[([^\]]+)\]\(https://anilist\.co/character/\d+/[^)]*\)', raw_desc)
    all_related = list(set(peop + md_related))

    return {
        "id": char['id'], 
        "name": name, 
        "origin": char.get('origin_media', 'Unknown'),
        "metadata": metadata, 
        "entities": {
            "related_characters": all_related,
            "locations": locs,
            "organizations": orgs
        },
        "traits": [], # Sera rempli par le script d'embeddings si besoin
        "popularity": {
            "favourites": char.get('favourites', 0),
            "rank": get_popularity_rank(char.get('favourites', 0))
        },
        "biography": biography, 
        "image": char.get('image', {}).get('large') if isinstance(char.get('image'), dict) else char.get('image')
    }

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(BASE_DIR, 'data', 'raw', 'raw_characters_db.json')
    output_file = os.path.join(BASE_DIR, 'data', 'processed', 'refined_characters.json')
    
    if not os.path.exists(input_file):
        print("Source non trouvée.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        chars_db = json.load(f)
    
    # Pour tester, on peut limiter le nombre si besoin, mais on va faire le total
    print(f"Raffinage de {len(chars_db)} personnages...")
    
    refined_db = []
    # On utilise tqdm pour voir la progression (car le NER prend du temps)
    for c in tqdm(chars_db):
        refined_db.append(refine_character(c))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(refined_db, f, indent=2, ensure_ascii=False)
    print("✅ Terminé !")

if __name__ == "__main__":
    main()
