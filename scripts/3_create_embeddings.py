import json
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading the clean database (clean_root_animes.json)...")
try:
    with open('../data/processed/clean_root_animes.json', 'r', encoding='utf-8') as f:
        clean_animes_db = json.load(f)
except FileNotFoundError:
    print("ERROR: The file 'clean_root_animes.json' was not found.")
    print("Please run Phase 1B (2_filter_franchise.py) first.")
    exit()

# --- Data Preparation ---
corpus = []
data_for_lookup = []  # NEW: This list will become your JSON

print("Preparing the corpus...")
for anime in clean_animes_db:
    # We make sure the fields exist
    if not anime.get('description') or not anime.get('tags'):
        continue

    # 1. Creation of the "Super-Text" for the AI
    title = anime['title']
    genres = ", ".join(anime.get('genres', []))
    tags = ", ".join(anime.get('tags', []))
    description = anime['description']

    # Amplified version (recommended)
    super_text = f"Themes: {tags}. {tags}. {tags}. Genres: {genres}. {genres}. Plot: {description}"
    corpus.append(super_text)

    # 2. Preparation of the data list for Phase 3
    # We save the title and popularity in the SAME ORDER
    data_for_lookup.append({
        "title": title,
        "popularity": anime.get('popularity', 0)  # Put 0 if popularity is missing
    })

print(f"{len(corpus)} valid animes to analyze.")

# --- Calculating Embeddings (Long Step) ---

# Load the best model
print("Loading the AI model (all-mpnet-base-v2)...")
model = SentenceTransformer('all-mpnet-base-v2')

# Generate the embeddings
print("Calculating vectors (Embeddings)... This is the long step.")
corpus_embeddings = model.encode(corpus, show_progress_bar=True)

# --- Saving the results ---

# 1. Save the vectors
np.save('../data/artifacts/clean_vectors.npy', corpus_embeddings)
print(f"Vectors saved in 'clean_vectors.npy'")

# 2. Save the lookup list (TITLE + POPULARITY)
# THIS IS THE FILE YOU WERE MISSING
with open('../data/artifacts/clean_data_for_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(data_for_lookup, f, indent=2)

print(f"Lookup data saved in 'clean_data_for_lookup.json'")
print("--- Phase 2 (Creating Embeddings) Complete! ---")