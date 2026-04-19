from fastapi import FastAPI, HTTPException
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Animetix Brain API")

# Chargement des données (On profite des 16 Go de RAM de HF)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "artifacts")

vectors = {}
lookup = {}

def load_brain():
    global vectors, lookup
    modes = ['anime', 'manga', 'char']
    for mode in modes:
        try:
            # Sur HF, on peut charger en RAM direct pour une vitesse maximale
            vectors[f"{mode}_thematic"] = np.load(os.path.join(DATA_PATH, f"{mode}_thematic_vectors.npy" if mode != 'char' else "char_vectors.npy"))
            lookup[mode] = json.load(open(os.path.join(DATA_PATH, f"{mode}_data_for_lookup.json"), encoding='utf-8'))
            print(f"✅ {mode} loaded in Brain.")
        except Exception as e:
            print(f"❌ Error loading {mode}: {e}")

@app.on_event("startup")
async def startup_event():
    load_brain()

class SimilarityRequest(BaseModel):
    mode: str  # 'anime', 'manga', 'char'
    secret_idx: int
    guess_idx: int

@app.post("/similarity")
async def get_similarity(req: SimilarityRequest):
    mode = req.mode
    if f"{mode}_thematic" not in vectors:
        raise HTTPException(status_code=404, detail="Mode not loaded")
    
    v = vectors[f"{mode}_thematic"]
    s_vec = v[req.secret_idx].reshape(1, -1)
    g_vec = v[req.guess_idx].reshape(1, -1)
    
    sim = float(cosine_similarity(s_vec, g_vec)[0][0])
    return {"similarity": sim}

@app.get("/health")
def health():
    return {"status": "alive"}
