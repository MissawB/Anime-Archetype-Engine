import numpy as np
import json
import os
import random
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from typing import Any, List, Optional, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# --- STRUCTURED OUTPUT MODELS ---
class ScenarioOutput(BaseModel):
    reasoning: str = Field(description="Réflexion étape par étape.")
    scenario: str = Field(description="Le synopsis final.")

class ExplanationOutput(BaseModel):
    reasoning: str = Field(description="Analyse thématique.")
    explanation: str = Field(description="Explication courte.")

# --- 1. ANIMETIX DATA SERVICE ---
class AnimetixService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnimetixService, cls).__new__(cls)
            cls._instance.data = {}
        return cls._instance

    def load_data(self, media_type):
        if media_type in self.data:
            return self.data[media_type]
                
        # --- v91 FIX : Détection robuste du chemin data ---
        # On part du dossier de ce fichier : backend/animetix/services.py
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        # On remonte de 2 niveaux pour arriver à la racine (Double_scenario_Project/)
        project_root = os.path.dirname(os.path.dirname(current_file_dir))
        
        # Log de debug pour vérifier où on cherche
        print(f"🔍 Searching data in root: {project_root}")
        
        base_path = project_root
        data_configs = {
            'Anime': {
                "thematic": 'data/artifacts/anime_thematic_vectors.npy',
                "plot": 'data/artifacts/anime_plot_vectors.npy',
                "vibe": 'data/artifacts/anime_vibe_vectors.npy',
                "lookup": 'data/artifacts/anime_data_for_lookup.json',
                "db": 'data/processed/clean_root_animes.json'
            },
            'Manga': {
                "thematic": 'data/artifacts/manga_thematic_vectors.npy',
                "plot": 'data/artifacts/manga_plot_vectors.npy',
                "vibe": 'data/artifacts/manga_vibe_vectors.npy',
                "lookup": 'data/artifacts/manga_data_for_lookup.json',
                "db": 'data/processed/clean_root_mangas.json'
            },
            'Character': {
                "thematic": 'data/artifacts/char_vectors.npy',
                "plot": 'data/artifacts/char_vectors.npy',
                "vibe": 'data/artifacts/char_vectors.npy',
                "lookup": 'data/artifacts/char_data_for_lookup.json',
                "db": 'data/processed/filtered_characters.json' # Updated to filtered
            }
        }
        
        config = data_configs.get(media_type)
        if not config: return None

        try:
            lookup_path = os.path.join(base_path, config["lookup"])
            db_path = os.path.join(base_path, config["db"])
            
            if not os.path.exists(lookup_path) or not os.path.exists(db_path):
                print(f"❌ Missing files in {base_path}/data/...")
                # DIAGNOSTIC : On liste ce que Django voit réellement
                try:
                    import glob
                    print(f"📂 Contenu de {base_path}: {os.listdir(base_path)}")
                    if os.path.exists(os.path.join(base_path, 'data')):
                        print(f"📂 Contenu de {base_path}/data: {os.listdir(os.path.join(base_path, 'data'))}")
                        for sub in ['artifacts', 'processed']:
                            p = os.path.join(base_path, 'data', sub)
                            if os.path.exists(p):
                                print(f"📂 Contenu de {p}: {os.listdir(p)}")
                except: pass

            self.data[media_type] = {
                "lookup": json.load(open(lookup_path, encoding='utf-8')),
                "db": json.load(open(db_path, encoding='utf-8')),
            }
            
            # On charge TOUJOURS les vecteurs localement en mmap (très léger en RAM) si possible
            # Cela permet aux modes Paradox et Undercover de fonctionner instantanément
            try:
                print(f"🧠 Loading vectors in memory-map mode for {media_type}...")
                self.data[media_type]["vectors_thematic"] = np.load(os.path.join(base_path, config["thematic"]), mmap_mode='r')
                self.data[media_type]["vectors_plot"] = np.load(os.path.join(base_path, config["plot"]), mmap_mode='r')
                self.data[media_type]["vectors_vibe"] = np.load(os.path.join(base_path, config["vibe"]), mmap_mode='r')
                print(f"✅ {media_type} vectors loaded.")
            except Exception as ve:
                print(f"⚠️ Warning: Could not load vectors for {media_type}: {ve}")
                # On ne bloque pas tout le service, on continue avec les JSON uniquement
            
            if os.getenv("BRAIN_API_URL"):
                print(f"🌐 Remote Brain enabled for LLM tasks.")

            d = self.data[media_type]
            # On standardise : 'title' est la clé de référence
            d["titles"] = []
            for item in d["lookup"]:
                t = item.get('title') or item.get('name')
                if t: d["titles"].append(t)
            
            d["title_to_index"] = {t: i for i, t in enumerate(d["titles"])}
            
            d["title_to_full_data"] = {}
            for item in d["db"]:
                t = item.get('title') or item.get('name')
                if t: d["title_to_full_data"][t] = item
            
            return d
        except Exception as e:
            print(f"Error loading {media_type} data: {e}"); return None

# --- 2. LANGCHAIN SERVICE (Micro-service Client) ---
class LangChainService:
    def __init__(self):
        self.brain_url = os.getenv("BRAIN_API_URL")

    def _generate_via_brain(self, prompt, system_prompt="You are a helpful assistant."):
        """Appelle Llama 3.2 via l'API Brain."""
        if not self.brain_url:
            return None
        try:
            response = requests.post(f"{self.brain_url}/generate", json={
                "prompt": prompt,
                "system_prompt": system_prompt
            }, timeout=30)
            if response.status_code == 200:
                return response.json()["text"]
        except Exception as e:
            print(f"Brain LLM Error: {e}")
        return None

    def _safe_json_parse(self, text, default_reasoning="Analyse Indisponible", default_scenario="..."):
        """Extrait les champs avec une tolérance maximale aux erreurs de format."""
        if not text:
            return {"reasoning": default_reasoning, "scenario": default_scenario}
        
        # 1. Tentative JSON Standard
        try:
            clean_json = text[text.find('{'):text.rfind('}')+1]
            # On répare les sauts de ligne internes
            clean_json = clean_json.replace('\n', ' ').replace('\r', ' ')
            parsed = json.loads(clean_json)
            r = parsed.get('reasoning') or parsed.get('explanation')
            s = parsed.get('scenario') or parsed.get('synopsis')
            if r and s: return {"reasoning": r, "scenario": s}
        except: pass

        # 2. Tentative par REGEX (plus robuste si guillemets manquants ou mal échappés)
        r_match = re.search(r'["\']?reasoning["\']?\s*:\s*["\'](.*?)["\'](?=,|$|\s*})', text, re.DOTALL | re.IGNORECASE)
        s_match = re.search(r'["\']?scenario["\']?\s*:\s*["\'](.*?)["\'](?=,|$|\s*})', text, re.DOTALL | re.IGNORECASE)
        
        reasoning = r_match.group(1) if r_match else ""
        scenario = s_match.group(1) if s_match else ""

        # 3. Tentative par détection de texte brut (si l'IA n'a pas mis de JSON du tout)
        if not reasoning or len(reasoning) < 5:
            if "parce que" in text.lower() or "commun" in text.lower():
                reasoning = text[:300].split('}')[0].strip(' "{}\n')
            else:
                reasoning = default_reasoning

        if not scenario or len(scenario) < 10:
            scenario = text.split('scenario')[-1].strip(' ":{}\n') if 'scenario' in text else text

        return {
            "reasoning": reasoning.replace('\\n', '\n').replace('\\"', '"').strip(),
            "scenario": scenario.replace('\\n', '\n').replace('\\"', '"').strip()
        }

    def generate_scenario_advanced(self, media_type, item_A, item_B, language):
        icon = "🦙"
        label_A = item_A.get('title') or item_A.get('name') or "Entité A"
        label_B = item_B.get('title') or item_B.get('name') or "Entité B"
        
        system_prompt = f"Tu es un expert Concept Creator spécialisé en {media_type}. Réponds UNIQUEMENT par un objet JSON."
        user_prompt = f"""
        MISSION : Fusionne l'univers de {label_A} et {label_B}.
        LANGUE : {language}. Pas de noms cités. 
        JSON STRICT: {{"reasoning": "logique {icon}", "scenario": "histoire"}}
        """
        res_text = self._generate_via_brain(user_prompt, system_prompt)
        return self._safe_json_parse(res_text, f"Analyse Llama {icon}", "Fusion en cours...")

    def generate_paradox_logic(self, media_type, item_A, item_B, item_I, language):
        """Analyse le lien entre A/B et pourquoi I est l'intrus."""
        icon = "🧩"
        label_A = item_A.get('title') or item_A.get('name')
        label_B = item_B.get('title') or item_B.get('name')
        label_I = item_I.get('title') or item_I.get('name')
        
        system_prompt = f"Tu es un expert Concept Creator. Ta mission est d'expliquer pourquoi '{label_I}' est l'intrus par rapport à '{label_A}' et '{label_B}'."
        user_prompt = f"""
        ANALYSE CES 3 ÉLÉMENTS :
        1. '{label_A}' : {item_A.get('description', '')[:200]}
        2. '{label_B}' : {item_B.get('description', '')[:200]}
        3. INTRUS : '{label_I}' : {item_I.get('description', '')[:200]}
        
        MISSION :
        - Dans "reasoning" : Explique spécifiquement pourquoi '{label_A}' et '{label_B}' sont similaires et pourquoi '{label_I}' est différent.
        - Dans "scenario" : Décris le point commun de '{label_A}' et '{label_B}' sans les nommer.
        
        RÉPONDS UNIQUEMENT AU FORMAT JSON :
        {{
            "reasoning": "Ton explication ici {icon}",
            "scenario": "Ton synopsis ici"
        }}
        """
        res_text = self._generate_via_brain(user_prompt, system_prompt)
        return self._safe_json_parse(res_text, f"L'IA n'a pas pu justifier ce choix. {icon}", res_text)

    def generate_undercover_clue(self, media_type, item_A, item_B, language):
        prompt = f"Donne un seul mot-clé thématique commun à ces deux {media_type} : '{item_A}' et '{item_B}'. Ne cite aucun nom."
        res = self._generate_via_brain(prompt)
        return res if res else "Mystère..."

    def explain_undercover(self, maj_titles, maj_tags, intruder_title, intruder_tags, language):
        icon = "🦙"
        prompt = f"""
        Groupe d'entités : {', '.join(maj_titles)}. 
        L'intrus : {intruder_title}.
        
        Explique brièvement pourquoi c'est l'intrus.
        Réponds UNIQUEMENT au format JSON suivant :
        {{
            "reasoning": "Analyse thématique {icon}",
            "explanation": "ton explication courte en {language}"
        }}
        """
        res_text = self._generate_via_brain(prompt, "Tu es un expert en analyse d'animés.")
        if res_text:
            try:
                json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            except: pass
        return {"reasoning": f"Erreur {icon}", "explanation": "Analyse non disponible."}
