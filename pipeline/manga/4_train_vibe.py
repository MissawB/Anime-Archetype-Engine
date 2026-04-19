import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

MODEL_NAME = 'all-mpnet-base-v2'
OUTPUT_PATH = '../../data/models/manga-vibe-model'
BATCH_SIZE = 16
EPOCHS = 2

print("--- Début du Fine-Tuning des Embeddings Manga (Vibe) ---")

try:
    with open('../../data/processed/clean_root_mangas.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Erreur : clean_root_mangas.json introuvable.")
    exit()

train_examples = []
for manga in data:
    title = manga['title']
    for review in manga.get('reviews', []):
        if len(review) > 20:
            train_examples.append(InputExample(texts=[review, title]))

if len(train_examples) < 10:
    print("Pas assez de critiques.")
    exit()

model = SentenceTransformer(MODEL_NAME)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=100,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

print(f"✅ Modèle fine-tuné Manga sauvegardé.")
