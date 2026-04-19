import requests
import json
import time
import os

url = 'https://graphql.anilist.co'

query = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    media(type: ANIME, sort: [POPULARITY_DESC]) {
      id
      format
      popularity
      title {
        romaji
        english
        native
      }
      description(asHtml: false)
      genres
      tags {
        name
        rank
      }
      startDate {
        year
      }
      coverImage {
        large
      }
      reviews(perPage: 20, sort: [RATING_DESC]) {
        nodes {
          summary
          body
          rating
        }
      }
      recommendations(perPage: 10, sort: [RATING_DESC]) {
        nodes {
          rating
          mediaRecommendation {
            title {
              romaji
            }
          }
        }
      }
      relations {
        edges {
          relationType
          node {
            id
            format
          }
        }
      }
    }
  }
}
"""

variables = {
    'page': 1,
    'perPage': 50
}

all_animes = []
has_next_page = True

print("🚀 Starting Anime Collection...")

while has_next_page:
    try:
        response = requests.post(url, json={'query': query, 'variables': variables}, timeout=30)
        
        if response.status_code == 200:
            data = response.json().get('data', {}).get('Page')
            if not data:
                print("⚠️ No data found in response.")
                break
                
            media = data.get('media', [])
            for anime in media:
                if anime.get('description'):
                    all_animes.append(anime)

            has_next_page = data['pageInfo']['hasNextPage']
            print(f"✅ Page {variables['page']} retrieved. Total animes: {len(all_animes)}")
            
            variables['page'] += 1
            time.sleep(0.7)
            
            # Limite pour le test ou pour éviter de tout prendre si on veut (Optionnel)
            if variables['page'] > 100: # On s'arrête à 5000 animes pour la v1
                break
        elif response.status_code == 429:
            print("⚠️ Rate limit. Sleeping...")
            time.sleep(60)
        else:
            print(f"❌ Error: {response.status_code}")
            break
    except Exception as e:
        print(f"❌ Exception: {e}")
        break

# Sauvegarde forcée
OUTPUT_FILE = '../../data/raw/raw_anilist_db.json'
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_animes, f, indent=2, ensure_ascii=False)

print(f"✅ Collection finished! Saved {len(all_animes)} animes to {OUTPUT_FILE}")
