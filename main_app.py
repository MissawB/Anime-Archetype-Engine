import numpy as np
import json
import random
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
SIMILARITY_THRESHOLD = 0.65

def initialize_ai():
    """Initializes and configures the Gemini AI model."""
    try:
        genai.configure(api_key=API_KEY_GEMINI)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("🤖 Creative Brain (Gemini AI) initialized.")
        return model
    except Exception as e:
        print(f"❌ Error during Gemini API configuration: {e}")
        print("Check your API key or internet connection.")
        exit()

def get_user_choices():
    """Gets media type and language choices from the user."""
    while True:
        mode = input("Analyze [A]nime or [M]anga? ").strip().upper()
        if mode == 'A':
            media_type = "Anime"
            paths = {
                "vectors": 'data/artifacts/anime_vectors.npy',
                "lookup": 'data/artifacts/anime_data_for_lookup.json',
                "db": 'data/processed/clean_root_animes.json'
            }
            break
        elif mode == 'M':
            media_type = "Manga"
            paths = {
                "vectors": 'data/artifacts/manga_vectors.npy',
                "lookup": 'data/artifacts/manga_data_for_lookup.json',
                "db": 'data/processed/clean_root_mangas.json'
            }
            break
        else:
            print("Invalid input. Please enter 'A' or 'M'.")

    while True:
        language = input("Choose the output language (e.g., English, French, Spanish): ").strip()
        if language:
            break
        else:
            print("Please enter a language.")
            
    return media_type, paths, language

def load_data(paths, media_type):
    """Loads all necessary data from files."""
    print(f"🧠 Loading the Analyst Brain ({media_type})...")
    try:
        vectors = np.load(paths["vectors"])
        with open(paths["lookup"], 'r', encoding='utf-8') as f:
            lookup_data = json.load(f)
        with open(paths["db"], 'r', encoding='utf-8') as f:
            full_db = json.load(f)
        print("✅ Data loaded.")
        return vectors, lookup_data, full_db
    except FileNotFoundError as e:
        print(f"❌ ERROR: Missing file: {e.filename}")
        print(f"Make sure you have run the preparation scripts for {media_type}s.")
        exit()

def get_similarity_score(idx_A, idx_B, vectors):
    """Calculates the cosine similarity score between two vectors."""
    vector_A = vectors[idx_A].reshape(1, -1)
    vector_B = vectors[idx_B].reshape(1, -1)
    score = cosine_similarity(vector_A, vector_B)
    return score[0][0]

def generate_scenario(item_A, item_B, media_type, language, ai_model):
    """Builds the prompt and calls the Gemini API to generate the archetypal scenario."""
    tags_A = ", ".join(item_A.get('tags', []))
    tags_B = ", ".join(item_B.get('tags', []))

    prompt = f"""
    [SYSTEM ROLE]
    You are a "Story Archetypist". Your mission is to analyze the two following synopses and extract their common archetypal plot.

    [{media_type.upper()} 1: INFORMATION]
    Title: {item_A['title']}
    Key Tags: {tags_A}
    Description: {item_A['description']}

    [{media_type.upper()} 2: INFORMATION]
    Title: {item_B['title']}
    Key Tags: {tags_B}
    Description: {item_B['description']}

    [YOUR MISSION]
    Write a single synopsis (about 150 words) that describes this fundamental plot, universal enough to describe each of the two {media_type}s individually.
    Do NOT use ANY character names or specific concepts unique to either work.
    IMPORTANT: Write the final synopsis in {language}.
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = ai_model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        return f"ERROR: AI call failed: {e}"

def main():
    """Main function to run the application."""
    ai_model = initialize_ai()
    media_type, paths, output_language = get_user_choices()
    all_vectors, lookup_data, full_db = load_data(paths, media_type)

    # Prepare data for fast lookups
    all_titles = [item['title'] for item in lookup_data]
    all_weights = [item['popularity'] for item in lookup_data]
    title_to_index = {title: i for i, title in enumerate(all_titles)}
    title_to_full_data = {item['title']: item for item in full_db}

    print("🚀 Application ready. Starting pair search...")

    while True:
        item_A_title, item_B_title = random.choices(all_titles, weights=all_weights, k=2)

        if item_A_title == item_B_title:
            continue

        idx_A = title_to_index[item_A_title]
        idx_B = title_to_index[item_B_title]
        
        score = get_similarity_score(idx_A, idx_B, all_vectors)

        if score >= SIMILARITY_THRESHOLD:
            print(f"\n--- COMPATIBILITY FOUND! (Score: {score:.2f}) ---")
            print(f"   {media_type} A: {item_A_title}")
            print(f"   {media_type} B: {item_B_title}")

            item_A_data = title_to_full_data[item_A_title]
            item_B_data = title_to_full_data[item_B_title]

            print(f"\n🤖 Generating archetypal scenario in {output_language}...")
            
            new_scenario = generate_scenario(item_A_data, item_B_data, media_type, output_language, ai_model)

            print(f"\n--- 🎬 ARCHETYPAL SCENARIO ({media_type} in {output_language}) ---")
            print(new_scenario)
            print("-----------------------------------")
            break

if __name__ == "__main__":
    main()
