import os
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import re

print("🚀 Initialisation du script d'embeddings Personnages...")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'filtered_characters.json')
OUTPUT_VECTORS_PATH = os.path.join(BASE_DIR, 'data', 'artifacts', 'char_vectors.npy')
OUTPUT_LOOKUP_PATH = os.path.join(BASE_DIR, 'data', 'artifacts', 'char_data_for_lookup.json')
FT_MODEL_PATH = os.path.join(BASE_DIR, 'data', 'models', 'character-vibe-model')

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    chars_db = json.load(f)

corpus = []
lookup_data = []

print("🔍 Encodage et préparation des données de jeu...")
for char in chars_db:
    # 1. Texte enrichi pour l'embedding
    name = char['name']
    bio = char['biography']
    meta = char.get('metadata', {})
    entities = char.get('entities', {})
    
    orgs = meta.get('affiliations', []) + entities.get('organizations', [])
    related = entities.get('related_characters', [])
    
    rich_text = f"Character: {name}. Orgs: {' '.join(orgs)}. Links: {' '.join(related)}. {bio}"
    corpus.append(rich_text)
    
    # 2. Données de comparaison pour Django
    # On normalise la taille en nombre pour le calcul
    h_str = meta.get('height', '0')
    h_val = 0
    h_match = re.search(r'(\d+)', str(h_str))
    if h_match: h_val = int(h_match.group(1))

    lookup_data.append({
        "title": name,
        "origin": char['origin'],
        "image": char['image'],
        "popularity": char.get('popularity', {}).get('favourites', 0),
        "organizations": [o.lower() for o in orgs],
        "related": [r.lower() for r in related],
        "height_cm": h_val
    })

print("🧠 Calcul des vecteurs...")
model = SentenceTransformer(FT_MODEL_PATH if os.path.exists(FT_MODEL_PATH) else 'all-mpnet-base-v2')
vectors = model.encode(corpus, show_progress_bar=True)

np.save(OUTPUT_VECTORS_PATH, vectors)
with open(OUTPUT_LOOKUP_PATH, 'w', encoding='utf-8') as f:
    json.dump(lookup_data, f, indent=2, ensure_ascii=False)

print("✅ Artifacts mis à jour avec données structurées !")
