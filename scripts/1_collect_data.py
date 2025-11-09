import requests
import json
import time

# The unique URL of the AniList API
url = 'https://graphql.anilist.co'

# The GraphQL query we defined above
query = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    # We sort by popularity to get the most relevant ones first
    media(type: ANIME, sort: [POPULARITY_DESC]) {
      id # ESSENTIAL for linking relationships
      format # ESSENTIAL for filtering (TV, MOVIE, OVA...)
      popularity # For weighted selection
      title {
        romaji
      }
      description(asHtml: false)
      genres
      tags {
        name
      }
      relations { # ESSENTIAL for filtering
        edges {
          relationType # (SEQUEL, PREQUEL, ADAPTATION, REMAKE...)
          node { # The linked anime
            id
            format
          }
        }
      }
    }
  }
}
"""

# Variables for our query (start at page 1, 50 animes per page)
variables = {
    'page': 1,
    'perPage': 50
}

# To store all our animes
all_animes = []
has_next_page = True

while has_next_page:
    # Send the POST request to the API
    response = requests.post(url, json={'query': query, 'variables': variables})

    if response.status_code == 200:
        data = response.json()['data']['Page']

        # Add the animes from this page to our list
        for anime in data['media']:
            # We clean the data a bit
            if anime['description']:  # If there is a description
                all_animes.append(anime)

        # Update for the next loop
        has_next_page = data['pageInfo']['hasNextPage']
        variables['page'] += 1
        print(f"Page {variables['page'] - 1} retrieved. Total animes: {len(all_animes)}")

        # VERY IMPORTANT: Respect the API limit
        # (90 requests per minute)
        time.sleep(0.7)  # A 0.7s pause is safe

    else:
        print(f"Error: {response.status_code}")
        break

# At the end, save everything to a file
with open('../data/raw/raw_anilist_db.json', 'w', encoding='utf-8') as f:
    json.dump(all_animes, f, indent=2, ensure_ascii=False)

print("Collection finished!")