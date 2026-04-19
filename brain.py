from fastapi import FastAPI, HTTPException
import numpy as np
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI(title="Animetix Brain API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "artifacts")

# Clients & Data
brain_data = {}
# On utilise l'Inference API gratuite de HF pour Llama 3.2
llm_client = InferenceClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    token=os.getenv("HF_TOKEN")
)

def load_vectors():
    global brain_data
    configs = {'anime': 'anime_thematic_vectors.npy', 'manga': 'manga_thematic_vectors.npy', 'char': 'char_vectors.npy'}
    for mode, filename in configs.items():
        path = os.path.join(DATA_PATH, filename)
        if os.path.exists(path):
            brain_data[mode] = np.load(path)
            print(f"✅ Brain: {mode} vectors loaded.")

@app.on_event("startup")
async def startup_event():
    load_vectors()

class SimilarityRequest(BaseModel):
    mode: str
    secret_idx: int
    guess_idx: int

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful assistant."

@app.post("/similarity")
async def get_similarity(req: SimilarityRequest):
    mode = req.mode.lower()[:4]
    if mode not in brain_data: raise HTTPException(status_code=404, detail="Mode not loaded")
    vecs = brain_data[mode]
    s_vec, g_vec = vecs[req.secret_idx].reshape(1, -1), vecs[req.guess_idx].reshape(1, -1)
    return {"similarity": float(cosine_similarity(s_vec, g_vec)[0][0])}

@app.post("/generate")
async def generate_text(req: GenerateRequest):
    try:
        # Appel à Llama 3.2 en production
        response = ""
        for message in llm_client.chat_completion(
            messages=[
                {"role": "system", "content": req.system_prompt},
                {"role": "user", "content": req.prompt}
            ],
            max_tokens=500,
            stream=True,
        ):
            response += message.choices[0].delta.content or ""
        return {"text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {
        "message": "Animetix Brain API is running",
        "endpoints": ["/health", "/similarity", "/generate"],
        "status": "online"
    }

@app.get("/health")
def health():
    return {"status": "online", "llm": "Llama-3.2-3B-Instruct via HF API"}
