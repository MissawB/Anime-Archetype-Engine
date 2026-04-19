import json
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
import re
import os

print("Loading the clean database (clean_root_mangas.json)...")
try:
    with open('../../data/processed/clean_root_mangas.json', 'r', encoding='utf-8') as f:
        clean_mangas_db = json.load(f)
except FileNotFoundError:
    print("ERROR: The file 'clean_root_mangas.json' was not found.")
    exit()

# --- Data Preparation ---
thematic_corpus = []
plot_corpus = []
vibe_corpus = [] 
data_for_lookup = []

def clean_plot_for_embedding(text, rare_words=None):
    if not text: return ""
    words = re.findall(r'\w+', text.lower())
    sentences = text.split('.')
    important_part = ". ".join(sentences[:2])
    boosted_terms = ""
    if rare_words:
        content_words = [w for w in words if w in rare_words]
        boosted_terms = " ".join(content_words)
    return f"{important_part}. {important_part}. {text} {boosted_terms}"

# Identification des mots rares
all_words = []
for manga in clean_mangas_db:
    if manga.get('description'):
        all_words.extend(re.findall(r'\w+', manga['description'].lower()))
word_counts = Counter(all_words)
total_docs = len(clean_mangas_db)
informative_min = max(2, total_docs * 0.01)
informative_max = total_docs * 0.15
rare_informative_words = {word for word, count in word_counts.items() if informative_min <= count <= informative_max}

print("Preparing the corpora...")
for manga in clean_mangas_db:
    if not manga.get('description') or not manga.get('tags'):
        continue

    title = manga['title']
    genres = ", ".join(manga.get('genres', []))
    tags = ", ".join(manga.get('tags', []))
    description = manga['description']
    reviews = " ".join(manga.get('reviews', []))
    recs = ", ".join(manga.get('recommendations', []))

    # 1. Thematic
    thematic_corpus.append(f"Themes: {tags}. Genres: {genres}. Fans also like: {recs}. {recs}.")

    # 2. Plot
    plot_text = clean_plot_for_embedding(description, rare_informative_words)
    plot_corpus.append(f"{plot_text} Similar plot to: {recs}")

    # 3. Vibe
    vibe_corpus.append(f"Fan opinions: {reviews}" if reviews else "No reviews available.")

    # 4. Lookup
    data_for_lookup.append({
        "title": title,
        "title_english": manga.get('title_english',""),
        "title_native": manga.get('title_native',""),
        "image": manga.get('image',""),
        "popularity": manga.get('popularity', 0)
    })

# --- Calculating Embeddings ---
FT_MODEL_PATH = '../../data/models/manga-vibe-model'
if os.path.exists(FT_MODEL_PATH):
    model = SentenceTransformer(FT_MODEL_PATH)
else:
    model = SentenceTransformer('all-mpnet-base-v2')

print("Calculating Thematic vectors...")
thematic_embeddings = model.encode(thematic_corpus, show_progress_bar=True)

print("Calculating Plot vectors...")
plot_embeddings = model.encode(plot_corpus, show_progress_bar=True)

print("Calculating Vibe vectors...")
vibe_embeddings = model.encode(vibe_corpus, show_progress_bar=True)

# --- Saving ---
np.save('../../data/artifacts/manga_thematic_vectors.npy', thematic_embeddings)
np.save('../../data/artifacts/manga_plot_vectors.npy', plot_embeddings)
np.save('../../data/artifacts/manga_vibe_vectors.npy', vibe_embeddings)

with open('../../data/artifacts/manga_data_for_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(data_for_lookup, f, indent=2, ensure_ascii=False)

print("--- Phase 2 Complete (Enriched Triple Embeddings) ---")
