import requests
import json
import time
import os

url = 'https://graphql.anilist.co'
FILE_PATH = '../../data/raw/raw_characters_db.json'

# Query enrichie pour récupérer des caractéristiques de jeu
query = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    characters(sort: [FAVOURITES_DESC]) {
      id
      name {
        userPreferred
        native
        full
      }
      image {
        large
      }
      description
      favourites
      gender
      age
      bloodType
      dateOfBirth {
        year
        month
        day
      }
      media {
        nodes {
          id
          title {
            romaji
            english
          }
          type
          format
          popularity
          meanScore
          isAdult
          genres
        }
      }
    }
  }
}
"""

# 1. Charger les données existantes
existing_characters = []
if os.path.exists(FILE_PATH):
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            existing_characters = json.load(f)
        print(f"📂 Base actuelle chargée : {len(existing_characters)} personnages.")
    except Exception as e:
        print(f"⚠️ Impossible de charger la base existante : {e}")

existing_ids = {char['id'] for char in existing_characters}

variables = {'page': 2316, 'perPage': 50}
has_next_page = True
new_added_count = 0
consecutive_known_count = 0
max_consecutive_known = 1000 # On s'arrête si on trouve 1000 persos à la suite déjà connus

print("🚀 Début de la collecte enrichie...")

try:
    while has_next_page:
        response = requests.post(url, json={'query': query, 'variables': variables})
        
        if response.status_code == 200:
            data = response.json()['data']['Page']
            chars_on_page = data['characters']
            
            for char in chars_on_page:
                if char['id'] not in existing_ids:
                    # On vérifie les données minimales
                    if char.get('description') and char.get('media') and char['media']['nodes']:
                        existing_characters.append(char)
                        existing_ids.add(char['id'])
                        new_added_count += 1
                        consecutive_known_count = 0
                else:
                    consecutive_known_count += 1

            has_next_page = data['pageInfo']['hasNextPage']
            print(f"Page {variables['page']} traitée. Nouveaux : {new_added_count} | Total final : {len(existing_characters)}")
            
            if consecutive_known_count >= max_consecutive_known:
                print(f"🛑 Arrêt : {consecutive_known_count} personnages déjà connus. La base est à jour.")
                break
                
            variables['page'] += 1
            time.sleep(0.7) # Respect du rate limit
        elif response.status_code == 429:
            print("⚠️ Rate limit atteint. Pause de 60s...")
            time.sleep(60)
        elif response.status_code >= 500:
            print(f"⚠️ Erreur serveur {response.status_code}. Sauvegarde partielle...")
            break
        else:
            print(f"❌ Erreur API {response.status_code}")
            break
            
except Exception as e:
    print(f"⚠️ Interruption : {e}")

# 2. Sauvegarder la base mise à jour
if new_added_count > 0:
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(existing_characters, f, indent=2, ensure_ascii=False)
    print(f"✅ Terminé ! {new_added_count} nouveaux personnages ajoutés (Total: {len(existing_characters)}).")
else:
    print("✅ Terminé ! Aucun nouveau personnage à ajouter.")
