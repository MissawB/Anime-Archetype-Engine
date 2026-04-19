from fastapi import FastAPI, HTTPException
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from functools import lru_cache

app = FastAPI(title="Animetix Brain API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "artifacts")

# Stockage des vecteurs en RAM (HF offre 16 Go)
brain_data = {}

def load_brain():
    global brain_data
    configs = {
        'anime': 'anime_thematic_vectors.npy',
        'manga': 'manga_thematic_vectors.npy',
        'char': 'char_vectors.npy'
    }
    for mode, filename in configs.items():
        path = os.path.join(DATA_PATH, filename)
        if os.path.exists(path):
            brain_data[mode] = np.load(path)
            print(f"✅ Brain: {mode} vectors loaded ({len(brain_data[mode])} entities)")
        else:
            print(f"⚠️ Brain: {path} not found.")

@app.on_event("startup")
async def startup_event():
    load_brain()

class SimilarityRequest(BaseModel):
    mode: str
    secret_idx: int
    guess_idx: int

@app.post("/similarity")
async def get_similarity(req: SimilarityRequest):
    mode = req.mode.lower()[:4]
    if mode not in brain_data:
        raise HTTPException(status_code=404, detail=f"Mode {mode} not loaded in brain")
    
    try:
        vecs = brain_data[mode]
        s_vec = vecs[req.secret_idx].reshape(1, -1)
        g_vec = vecs[req.guess_idx].reshape(1, -1)
        sim = float(cosine_similarity(s_vec, g_vec)[0][0])
        return {"similarity": sim}
    except IndexError:
        raise HTTPException(status_code=400, detail="Index out of range")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "online", "loaded_modes": list(brain_data.keys())}
