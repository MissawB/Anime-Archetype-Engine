from dagster import asset, Definitions
import os
import subprocess
import sys

# Chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = BASE_DIR # Comme dagster_app.py est dans /pipeline

def run_python_script(script_name, category):
    script_path = os.path.join(PIPELINE_DIR, category, script_name)
    if not os.path.exists(script_path):
        return f"⚠️ Skip: {script_name} introuvable."
        
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Erreur dans {script_name}:\n{result.stderr}")
    return result.stdout

# --- 🎭 PIPELINE : CHARACTERS ---

@asset(group_name="characters")
def raw_characters():
    """Collecte les données brutes de l'API AniList (Characters)"""
    return run_python_script("1_ingest_characters.py", "characters")

@asset(deps=[raw_characters], group_name="characters")
def refined_characters():
    """Transformation NLP & IA des personnages"""
    return run_python_script("2_refine_characters.py", "characters")

@asset(deps=[refined_characters], group_name="characters")
def filtered_characters():
    """Filtrage qualité des personnages"""
    return run_python_script("3_filter_characters.py", "characters")

@asset(deps=[filtered_characters], group_name="characters")
def trained_characters_model():
    """Fine-tuning du modèle Vibe (Characters)"""
    return run_python_script("4_train_vibe.py", "characters")

@asset(deps=[trained_characters_model], group_name="characters")
def character_artifacts():
    """Génération des vecteurs et lookup JSON (Characters)"""
    return run_python_script("5_vectorize_characters.py", "characters")


# --- 📺 PIPELINE : ANIME ---

@asset(group_name="anime")
def raw_anime():
    """Collecte API AniList (Anime)"""
    return run_python_script("1_ingest_anime.py", "anime")

@asset(deps=[raw_anime], group_name="anime")
def filtered_anime():
    """Filtrage des franchises et doublons (Anime)"""
    return run_python_script("3_filter_anime.py", "anime")

@asset(deps=[filtered_anime], group_name="anime")
def trained_anime_model():
    """Fine-tuning du modèle Vibe (Anime)"""
    return run_python_script("4_train_vibe.py", "anime")

@asset(deps=[trained_anime_model], group_name="anime")
def anime_artifacts():
    """Génération des vecteurs thématiques et plot (Anime)"""
    return run_python_script("5_vectorize_anime.py", "anime")


# --- 📖 PIPELINE : MANGA ---

@asset(group_name="manga")
def raw_manga():
    """Collecte API AniList (Manga)"""
    return run_python_script("1_ingest_manga.py", "manga")

@asset(deps=[raw_manga], group_name="manga")
def filtered_manga():
    """Filtrage des franchises (Manga)"""
    return run_python_script("3_filter_manga.py", "manga")

@asset(deps=[filtered_manga], group_name="manga")
def trained_manga_model():
    """Fine-tuning du modèle Vibe (Manga)"""
    return run_python_script("4_train_vibe.py", "manga")

@asset(deps=[trained_manga_model], group_name="manga")
def manga_artifacts():
    """Génération des vecteurs (Manga)"""
    return run_python_script("5_vectorize_manga.py", "manga")


# --- DEFINITIONS ---

defs = Definitions(
    assets=[
        # Characters
        raw_characters, refined_characters, filtered_characters, 
        trained_characters_model, character_artifacts,
        # Anime
        raw_anime, filtered_anime, trained_anime_model, anime_artifacts,
        # Manga
        raw_manga, filtered_manga, trained_manga_model, manga_artifacts
    ]
)
