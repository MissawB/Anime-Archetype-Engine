import numpy as np
import json
import random
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
SIMILARITY_THRESHOLD = 0.65  # Your "close enough" threshold (adjust if needed)

# --- 1. AI CONFIGURATION (PHASE 3) ---
try:
    genai.configure(api_key=API_KEY_GEMINI)
    ai_model = genai.GenerativeModel('gemini-1.5-flash')  # Fast and efficient
    print("🤖 Creative Brain (Gemini AI) initialized.")
except Exception as e:
    print(f"❌ Error during Gemini API configuration: {e}")
    print("Check your API key or internet connection.")
    exit()

# --- 2. LOADING DATA (PHASE 2) ---
print("🧠 Loading the Analyst Brain (Vectors & Data)...")
try:
    # Load the vectors (the "coordinates")
    all_vectors = np.load('data/artifacts/clean_vectors.npy')

    # Load the data for selection (title + popularity)
    with open('data/artifacts/clean_data_for_lookup.json', 'r', encoding='utf-8') as f:
        all_anime_data_lookup = json.load(f)

    # Load the complete database for prompts (synopsis, tags...)
    with open('data/processed/clean_root_animes.json', 'r', encoding='utf-8') as f:
        clean_animes_db = json.load(f)

    print("✅ Data loaded.")
except FileNotFoundError as e:
    print(f"❌ ERROR: Missing file: {e.filename}")
    print(
        "Please ensure that 'clean_vectors.npy', 'clean_data_for_lookup.json', and 'clean_root_animes.json' are in the same directory.")
    exit()

# --- 3. PREPARING DATA FOR THE APP ---

# Preparation for weighted selection
all_titles = [anime['title'] for anime in all_anime_data_lookup]
all_weights = [anime['popularity'] for anime in all_anime_data_lookup]

# Preparation for instant access to vectors
title_to_index = {title: i for i, title in enumerate(all_titles)}

# Preparation for instant access to full data (for prompts)
title_to_full_data = {anime['title']: anime for anime in clean_animes_db}

print("🚀 Application ready. Starting pair search...")


# --- 4. APPLICATION FUNCTIONS ---

def get_similarity_score(title_A, title_B):
    """Calculates the similarity score between two animes."""
    idx_A = title_to_index[title_A]
    idx_B = title_to_index[title_B]
    vector_A = all_vectors[idx_A].reshape(1, -1)
    vector_B = all_vectors[idx_B].reshape(1, -1)
    score = cosine_similarity(vector_A, vector_B)
    return score[0][0]


def generate_scenario(anime_A, anime_B):
    """
    Builds the prompt and calls the Gemini API to generate
    the ARCHETYPAL SCENARIO (non-crossover).
    """

    # Formatting tags for clean display
    tags_A = ", ".join(anime_A.get('tags', []))
    tags_B = ", ".join(anime_B.get('tags', []))

    # --- START OF NEW PROMPT ---
    prompt = f"""
    [SYSTEM ROLE]
    You are a "Story Archetypist". Your mission is NOT to create a crossover or merge universes.
    Your mission is to analyze the following two synopses and extract their **archetypal plot**: the central conflict and the common theme that unites them.

    [ANIME 1: INFORMATION]
    Title: {anime_A['title']}
    Key Tags: {tags_A}
    Description: {anime_A['description']}

    [ANIME 2: INFORMATION]
    Title: {anime_B['title']}
    Key Tags: {tags_B}
    Description: {anime_B['description']}

    [YOUR MISSION]
    Write a single synopsis (about 150 words) that describes this fundamental plot. This synopsis must be universal enough to describe *each* of the two animes individually.

    IMPORTANT: Do NOT use ANY character names, places, or specific concepts (e.g., "Naruto", "Dragon Ball", "Geass", etc.) unique to either anime. Focus only on the common themes and parts of the universe.
    """
    # --- END OF NEW PROMPT ---

    try:
        # Configuration to avoid security blocks
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        response = ai_model.generate_content(prompt, safety_settings=safety_settings)
        return response.text

    except Exception as e:
        # Improved error handling
        try:
            feedback = e.response.prompt_feedback
            return f"ERROR: Generation was blocked. Reason: {feedback}"
        except AttributeError:
            return f"ERROR: Problem during the call to the AI: {e}"


# --- 5. MAIN APPLICATION LOOP ---
while True:
    # 1. WEIGHTED SELECTION
    pair = random.choices(all_titles, weights=all_weights, k=2)
    anime_A_title = pair[0]
    anime_B_title = pair[1]

    if anime_A_title == anime_B_title:
        continue

    # 2. SIMILARITY CALCULATION
    score = get_similarity_score(anime_A_title, anime_B_title)

    # 3. VERIFICATION
    if score >= SIMILARITY_THRESHOLD:
        print(f"\n--- COMPATIBILITY FOUND! (Score: {score:.2f}) ---")
        print(f"   Anime A: {anime_A_title}")
        print(f"   Anime B: {anime_B_title}")

        # 4. RETRIEVING FULL DATA
        anime_A_data = title_to_full_data[anime_A_title]
        anime_B_data = title_to_full_data[anime_B_title]

        print("\n🤖 Generating archetypal scenario...")

        # 5. CALL TO AI (PHASE 3)
        new_scenario = generate_scenario(anime_A_data, anime_B_data)

        print("\n--- 🎬 ARCHETYPAL SCENARIO ---")
        print(new_scenario)
        print("-----------------------------------")

        break  # We stop the script after finding a result