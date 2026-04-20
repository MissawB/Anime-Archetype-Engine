import json
import os
import re
import logging
import sys

# --- v89.2 FIX : Robuste Auth & Silence ---
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

try:
    from dotenv import load_dotenv
    # Remonte de 3 niveaux : pipeline/characters/4_train_vibe.py -> pipeline/characters -> pipeline -> racine
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'))
    token = os.getenv("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
except Exception as e:
    # On continue même si l'auth échoue, le message sera masqué par le logger.error
    pass

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Configuration
MODEL_NAME = 'all-mpnet-base-v2'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'filtered_characters.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'models', 'character-vibe-model')
BATCH_SIZE = 16
EPOCHS = 3

print(f"--- Début du Fine-Tuning des Embeddings Personnages (Vibe) ---")

if not os.path.exists(INPUT_FILE):
    print(f"Erreur : {INPUT_FILE} introuvable. Lancez d'abord le filtrage.")
    exit()

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

train_examples = []
for char in data:
    title = char.get('title') or char.get('name') or "Unknown"
    bio = char.get('biography', '')
    meta = char.get('metadata', {})
    entities = char.get('entities', {})
    
    orgs = meta.get('affiliations', []) + entities.get('organizations', [])
    traits = char.get('traits', [])
    if orgs or traits:
        attributes_text = f"{', '.join(traits)}. Affiliations: {', '.join(orgs)}."
        if len(attributes_text) > 10:
            train_examples.append(InputExample(texts=[attributes_text, title]))
            
    if len(bio) > 50:
        bio_snippet = bio[:300]
        train_examples.append(InputExample(texts=[bio_snippet, title]))

if len(train_examples) < 20:
    print("Pas assez d'exemples d'entraînement.")
    exit()

print(f"Nombre d'exemples d'entraînement générés : {len(train_examples)}")

print(f"Chargement du modèle : {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

print(f"Entraînement en cours...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=100,
    output_path=OUTPUT_PATH,
    show_progress_bar=True
)

print(f"✅ Modèle fine-tuné sauvegardé dans : {OUTPUT_PATH}")
