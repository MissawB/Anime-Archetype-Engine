import numpy as np
import json
import os
import random
import torch
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
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
            
            if not os.path.exists(lookup_path):
                print(f"❌ Missing lookup file: {lookup_path}")
            if not os.path.exists(db_path):
                print(f"❌ Missing DB file: {db_path}")

            self.data[media_type] = {
                "lookup": json.load(open(lookup_path, encoding='utf-8')),
                "db": json.load(open(db_path, encoding='utf-8')),
            }
            
            # OPTIMISATION HYBRIDE : On ne charge les vecteurs QUE si on n'utilise pas l'API Brain
            if not os.getenv("BRAIN_API_URL"):
                print(f"🧠 Loading vectors locally for {media_type}...")
                self.data[media_type]["vectors_thematic"] = np.load(os.path.join(base_path, config["thematic"]), mmap_mode='r')
                self.data[media_type]["vectors_plot"] = np.load(os.path.join(base_path, config["plot"]), mmap_mode='r')
                self.data[media_type]["vectors_vibe"] = np.load(os.path.join(base_path, config["vibe"]), mmap_mode='r')
            else:
                print(f"🌐 Using Remote Brain for {media_type} vectors.")

            d = self.data[media_type]
            d["titles"] = [item['title'] for item in d["lookup"]]
            d["title_to_index"] = {t: i for i, t in enumerate(d["titles"])}
            d["title_to_full_data"] = {item['title']: item for item in d["db"]}
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

    def generate_scenario_advanced(self, media_type, item_A, item_B, language):
        icon = "🦙" # Llama est maintenant le standard
        
        # Préparation du contexte (identique au précédent)
        if media_type == 'Character':
            label_A, label_B = item_A['title'], item_B['title']
            context_A, context_B = item_A.get('description', ''), item_B.get('description', '')
        else:
            label_A, label_B = item_A['title'], item_B['title']
            context_A = f"{item_A.get('description', '')}"
            context_B = f"{item_B.get('description', '')}"

        system_prompt = f"Tu es un expert Concept Creator spécialisé en {media_type}. Réponds UNIQUEMENT en JSON."
        user_prompt = f"""
        MISSION : Fusionne ces deux entités :
        1. {label_A} : {context_A[:800]}
        2. {label_B} : {context_B[:800]}
        
        Format JSON STRICT :
        {{
            "reasoning": "2 points techniques + {icon}",
            "scenario": "Synopsis en {language} (sans citer les noms)."
        }}
        """
        
        res_text = self._generate_via_brain(user_prompt, system_prompt)
        if res_text:
            try:
                # On tente de parser le JSON renvoyé par Llama
                import json
                # Nettoyage si Llama ajoute du texte autour du JSON
                json_match = re.search(r'\{.*\}', res_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
            except: pass
            return {"reasoning": f"Llama 3.2 {icon}", "scenario": res_text}
            
        return {"reasoning": "Fallback", "scenario": "L'IA est indisponible."}

    def generate_undercover_clue(self, media_type, item_A, item_B, language):
        prompt = f"Donne un seul mot-clé thématique commun à ces deux {media_type} : '{item_A}' et '{item_B}'. Ne cite aucun nom."
        res = self._generate_via_brain(prompt)
        return res if res else "Mystère..."

    def explain_undercover(self, maj_titles, maj_tags, intruder_title, intruder_tags, language):
        parser = JsonOutputParser(pydantic_object=ExplanationOutput)
        icon = self._get_icon()
        prompt = ChatPromptTemplate.from_template("""
        Groupe: {maj_titles}. Intrus: {intruder_title}. 
        Réponds UNIQUEMENT avec un objet JSON au format suivant:
        {{
            "reasoning": "Analyse thématique ({icon})",
            "explanation": "ton explication courte en {language}"
        }}
        """)
        chain = prompt | self.llm | parser
        try:
            return self._fix_nested_json(chain.invoke({
                "maj_titles": ", ".join(maj_titles), "intruder_title": intruder_title,
                "language": language, "icon": icon
            }))
        except: return {"reasoning": f"Error {icon}", "explanation": "Analyse non disponible."}
