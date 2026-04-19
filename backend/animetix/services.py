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
                
        base_path = settings.BASE_DIR.parent
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
            # On charge toujours les JSON (nécessaires pour les titres et images)
            self.data[media_type] = {
                "lookup": json.load(open(os.path.join(base_path, config["lookup"]), encoding='utf-8')),
                "db": json.load(open(os.path.join(base_path, config["db"]), encoding='utf-8')),
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

# --- 2. LANGCHAIN SERVICE (Hybrid Resilience) ---
class LangChainService:
    _llm_instance = None
    _model_type = "gemini" 

    def __init__(self):
        if LangChainService._llm_instance is None:
            try:
                print(f"🚀 Initialisation de Llama 3.2 via Ollama...")
                LangChainService._llm_instance = ChatOllama(model="llama3.2:3b", temperature=0.7)
                LangChainService._llm_instance.invoke("test")
                LangChainService._model_type = "llama"
                print("✅ Llama 3.2 Loaded.")
            except Exception as e:
                print(f"⚠️ Ollama failed: {e}. Falling back to Gemini API.")
                LangChainService._llm_instance = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=os.getenv("GEMINI_API_KEY")
                )
                LangChainService._model_type = "gemini"
        self.llm = LangChainService._llm_instance

    def _get_icon(self):
        return "🦙" if LangChainService._model_type == "llama" else "✨"

    def _fix_nested_json(self, data):
        if not isinstance(data, dict): return data
        if "properties" in data: data = data["properties"]
        for key in ["reasoning", "scenario", "explanation"]:
            if key in data and isinstance(data[key], list):
                data[key] = " ".join([str(item) for item in data[key]])
        return data

    def generate_scenario_advanced(self, media_type, item_A, item_B, language, player_history="Aucun"):
        parser = JsonOutputParser(pydantic_object=ScenarioOutput)
        icon = self._get_icon()
        
        if media_type == 'Character':
            task = f"Crée une rencontre ou une fusion entre ces deux personnages (Traits, Rôles, Capacités)"
            label_A = f"Perso 1: {item_A['title']} ({item_A.get('origin', 'Inconnu')})"
            label_B = f"Perso 2: {item_B['title']} ({item_B.get('origin', 'Inconnu')})"
            context_A = item_A.get('description', '')
            context_B = item_B.get('description', '')
        else:
            task = "Fusionne ces deux univers (Thèmes, Lore, Ambiance)"
            label_A = f"Œuvre A: {item_A['title']}"
            label_B = f"Œuvre B: {item_B['title']}"
            context_A = f"{item_A.get('description', '')} (Avis: {' '.join(item_A.get('reviews', []))})"
            context_B = f"{item_B.get('description', '')} (Avis: {' '.join(item_B.get('reviews', []))})"

        prompt = ChatPromptTemplate.from_template("""
        Tu es un expert Concept Creator spécialisé en {media_type}. 
        MISSION : {task}.
        
        1. {label_A} : {context_A}
        2. {label_B} : {context_B}
        
        CONSIGNES DE RÉPONSE (Format JSON STRICT) :
        - "reasoning" : Liste UNIQUEMENT les 2 points techniques précis de la fusion (ex: "Pouvoirs psy + École militaire"). Ajoute l'icône {icon} à la fin.
        - "scenario" : Rédige le synopsis en {language}. INTERDICTION de citer les noms/titres. Sois évocateur.
        
        Réponds UNIQUEMENT avec l'objet JSON.
        {{
            "reasoning": "...",
            "scenario": "..."
        }}
        """)
        
        chain = prompt | self.llm | parser
        try:
            return self._fix_nested_json(chain.invoke({
                "media_type": media_type, "task": task,
                "label_A": label_A, "context_A": context_A[:1200],
                "label_B": label_B, "context_B": context_B[:1200],
                "language": language, "icon": icon
            }))
        except Exception as e:
            print(f"LLM Error: {e}")
            return {"reasoning": f"Échec IA {icon}", "scenario": "L'IA a fait une erreur. Réessayez."}

    def generate_undercover_clue(self, media_type, item_A, item_B, language):
        icon = self._get_icon()
        if media_type == 'Character':
            prompt_text = f"Donne un seul mot-clé thématique (trait, rôle, élément visuel) commun à ces deux personnages : '{item_A}' et '{item_B}'. Ne cite aucun nom."
        else:
            prompt_text = f"Donne un seul mot-clé thématique commun à ces deux œuvres : '{item_A}' et '{item_B}'. Ne cite aucun titre."
            
        try: 
            res = self.llm.invoke(prompt_text)
            return res.content if hasattr(res, 'content') else str(res)
        except: return "Mystère..."

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
