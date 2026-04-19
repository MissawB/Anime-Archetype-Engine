import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Configuration
MODEL_NAME = 'all-mpnet-base-v2'
OUTPUT_PATH = '../../data/models/anime-vibe-model'
BATCH_SIZE = 16
EPOCHS = 2

print("--- Début du Fine-Tuning des Embeddings (Vibe) ---")

# 1. Chargement des données
try:
    with open('../../data/processed/clean_root_animes.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Erreur : clean_root_animes.json introuvable.")
    exit()

# 2. Préparation des exemples d'entraînement
train_examples = []
for anime in data:
    title = anime['title']
    # On crée des paires (Critique, Titre) pour que le modèle lie le ressenti au sujet
    for review in anime.get('reviews', []):
        if len(review) > 20:
            train_examples.append(InputExample(texts=[review, title]))

if len(train_examples) < 10:
    print("Pas assez de critiques pour un fine-tuning efficace.")
    exit()

print(f"Nombre d'exemples d'entraînement : {len(train_examples)}")

# 3. Chargement du modèle de base
model = SentenceTransformer(MODEL_NAME)

# 4. Configuration de l'entraînement
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# 5. Entraînement
print(f"Entraînement en cours pour {EPOCHS} époques...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=100,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

print(f"✅ Modèle fine-tuné sauvegardé dans : {OUTPUT_PATH}")
