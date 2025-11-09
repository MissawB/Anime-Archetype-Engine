import numpy as np
import json
import random
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION ---
API_KEY_GEMINI = os.getenv("GEMINI_API_KEY")
SIMILARITY_THRESHOLD = 0.65
genai.configure(api_key=API_KEY_GEMINI)
# Use a stable model
ai_model = genai.GenerativeModel('gemini-2.5-flash')
print("🤖 Creative Brain (Gemini AI) initialized.")

# --- 2. MODE & LANGUAGE SELECTION ---
while True:
    mode = input("Analyze [A]nime or [M]anga? ").strip().upper()
    if mode == 'A':
        MEDIA_TYPE_STR = "Anime"
        VECTORS_PATH = 'data/artifacts/anime_vectors.npy'
        LOOKUP_PATH = 'data/artifacts/anime_data_for_lookup.json'
        DB_PATH = 'data/processed/clean_root_animes.json'
        break
    elif mode == 'M':
        MEDIA_TYPE_STR = "Manga"
        VECTORS_PATH = 'data/artifacts/manga_vectors.npy'
        LOOKUP_PATH = 'data/artifacts/manga_data_for_lookup.json'
        DB_PATH = 'data/processed/clean_root_mangas.json'
        break
    else:
        print("Invalid input. Please enter 'A' or 'M'.")

# NEW: Language Selection
while True:
    output_language = input("Choose the output language (e.g., English, French, Spanish): ").strip()
    if output_language:  # Simple validation to ensure it's not empty
        break
    else:
        print("Please enter a language.")


print(f"🧠 Loading the Analyst Brain ({MEDIA_TYPE_STR})...")

# --- 3. DYNAMIC DATA LOADING ---
try:
    all_vectors = np.load(VECTORS_PATH)
    with open(LOOKUP_PATH, 'r', encoding='utf-8') as f:
        all_media_data_lookup = json.load(f)
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        clean_media_db = json.load(f)
    print("✅ Data loaded.")
except FileNotFoundError as e:
    print(f"❌ ERROR: Missing file: {e.filename}")
    print(f"Make sure you have run the preparation scripts for {MEDIA_TYPE_STR}s.")
    exit()

# --- 4. DATA PREPARATION ---
all_titles = [item['title'] for item in all_media_data_lookup]
all_weights = [item['popularity'] for item in all_media_data_lookup]
title_to_index = {title: i for i, title in enumerate(all_titles)}
title_to_full_data = {item['title']: item for item in clean_media_db}

print("🚀 Application ready. Starting pair search...")


# --- 5. FUNCTIONS (Prompt Updated) ---

def get_similarity_score(title_A, title_B):
    # This function is unchanged
    idx_A = title_to_index[title_A]
    idx_B = title_to_index[title_B]
    vector_A = all_vectors[idx_A].reshape(1, -1)
    vector_B = all_vectors[idx_B].reshape(1, -1)
    score = cosine_similarity(vector_A, vector_B)
    return score[0][0]


def generate_scenario(item_A, item_B, media_type, language):  # Added language
    """
    Builds the prompt and calls the Gemini API to generate
    the ARCHETYPAL SCENARIO.
    """
    tags_A = ", ".join(item_A.get('tags', []))
    tags_B = ", ".join(item_B.get('tags', []))

    # The prompt is now dynamic!
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
    Write a single synopsis (about 100 words) that describes this fundamental plot, universal enough to describe each of the two {media_type}s individually.
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
        try:
            feedback = e.response.prompt_feedback
            return f"ERROR: Generation was blocked. Reason: {feedback}"
        except AttributeError:
            return f"ERROR: Problem during the call to the AI: {e}"


# --- 6. MAIN LOOP (Call Updated) ---
while True:
    pair = random.choices(all_titles, weights=all_weights, k=2)
    item_A_title = pair[0]
    item_B_title = pair[1]

    if item_A_title == item_B_title:
        continue

    score = get_similarity_score(item_A_title, item_B_title)

    if score >= SIMILARITY_THRESHOLD:
        print(f"\n--- COMPATIBILITY FOUND! (Score: {score:.2f}) ---")
        print(f"   {MEDIA_TYPE_STR} A: {item_A_title}")
        print(f"   {MEDIA_TYPE_STR} B: {item_B_title}")

        item_A_data = title_to_full_data[item_A_title]
        item_B_data = title_to_full_data[item_B_title]

        print(f"\n🤖 Generating archetypal scenario in {output_language}...")

        # Pass the media type and language to the function!
        new_scenario = generate_scenario(item_A_data, item_B_data, MEDIA_TYPE_STR, output_language)

        print(f"\n--- 🎬 ARCHETYPAL SCENARIO ({MEDIA_TYPE_STR} in {output_language}) ---")
        print(new_scenario)
        print("-----------------------------------")

        break