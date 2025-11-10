import re
import json


def clean_description(text):
    """
    Clean the anime description by removing HTML tags,
    """
    if not text:
        return ""  # Return empty string if text is None or empty

    # 1. Delete HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2. Delete the part "(Source:...)"
    text = text.split('(Source:')[0].strip()

    # 3. Delete the part "Notes:"
    text = text.split('Notes:')[0].strip()

    # 4. Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("Loading the raw database...")
with open('../../data/raw/raw_anilist_db.json', 'r', encoding='utf-8') as f:
    all_animes = json.load(f)

# 1. Create a dictionary for quick access by ID
# (We only keep those that have a description)
animes_map = {
    anime['id']: anime
    for anime in all_animes
    if anime.get('description')
}

print(f"{len(animes_map)} animes with description loaded.")

# 2. Identify all animes that are NOT "roots"
# We create a "set" to store the IDs to exclude.
non_root_ids = set()

# Relationship types that "disqualify" an anime
RELATIONS_TO_EXCLUDE = [
    'PREQUEL',
    'REMAKE',
    'ALTERNATIVE_SETTING',
    'ALTERNATIVE_VERSION'
]

for anime_id, anime in animes_map.items():
    # A. Filter by format: We only want main TV series.
    # We exclude movies, OVAs, single episodes, etc.
    if anime['format'] not in ['TV', 'TV_SHORT']:
        non_root_ids.add(anime_id)
        continue  # No need to go further

    # B. Analyze relationships to find "children"
    for edge in anime['relations']['edges']:
        relation_type = edge['relationType']

        # If this anime IS a sequel/prequel/etc... of another...
        if relation_type in RELATIONS_TO_EXCLUDE:
            # ...it is NOT a root. We exclude it.
            non_root_ids.add(anime_id)
            break  # We can stop checking the relationships of this anime

print(f"{len(non_root_ids)} animes identified as non-roots (sequels, OVAs, etc.)")

# 3. Build the final and clean database
clean_root_animes = []
for anime_id, anime in animes_map.items():
    if anime_id not in non_root_ids:
        TAG_RELEVANCE_THRESHOLD = 70

        clean_tags = []
        if 'tags' in anime and anime['tags']:
            for tag in anime['tags']:
                # We only keep tags with a rank above the threshold
                if tag.get('rank') and tag['rank'] >= TAG_RELEVANCE_THRESHOLD:
                    clean_tags.append(tag['name'])

        #  Clean the description
        clean_desc = clean_description(anime.get('description'))

        # We only keep the fields useful for what follows
        clean_data = {
            'title': anime['title']['romaji'],
            'title_english': anime['title']['english'],
            'description': clean_desc,
            'genres': anime['genres'],
            'tags': clean_tags,
            'popularity': anime['popularity']
        }
        clean_root_animes.append(clean_data)

# 4. Save the clean database
with open('../../data/processed/clean_root_animes.json', 'w', encoding='utf-8') as f:
    json.dump(clean_root_animes, f, indent=2, ensure_ascii=False)

print(f"--- Phase 1B (Filtering) Complete! ---")
print(f"Clean database saved: {len(clean_root_animes)} 'root' animes.")