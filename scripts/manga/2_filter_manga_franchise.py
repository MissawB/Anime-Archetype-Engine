import json

print("Loading the raw database...")
with open('../../data/raw/raw_anilist_manga_db.json', 'r', encoding='utf-8') as f:
    all_mangas = json.load(f)

# 1. Create a dictionary for quick access by ID
# (We only keep those that have a description)
mangas_map = {
    manga['id']: manga
    for manga in all_mangas
    if manga.get('description')
}

print(f"{len(mangas_map)} mangas with description loaded.")

# 2. Identify all mangas that are NOT "roots"
# We create a "set" to store the IDs to exclude.
non_root_ids = set()

# Relationship types that "disqualify" a manga
RELATIONS_TO_EXCLUDE = [
    'SEQUEL',
    'PREQUEL',
    'SPIN_OFF',
    'REMAKE',
    'ALTERNATIVE_SETTING',
    'ALTERNATIVE_VERSION'
]

for manga_id, manga in mangas_map.items():
    # A. Filter by format: We only want main manga series.
    # We exclude novels, one-shots, etc.
    if manga['format'] not in ['MANGA']:
        non_root_ids.add(manga_id)
        continue  # No need to go further

    # B. Analyze relationships to find "children"
    for edge in manga['relations']['edges']:
        relation_type = edge['relationType']

        # If this manga IS a sequel/prequel/etc... of another...
        if relation_type in RELATIONS_TO_EXCLUDE:
            # ...it is NOT a root. We exclude it.
            non_root_ids.add(manga_id)
            break  # We can stop checking the relationships of this manga

print(f"{len(non_root_ids)} mangas identified as non-roots (sequels, spin-offs, etc.)")

# 3. Build the final and clean database
clean_root_mangas = []
for manga_id, manga in mangas_map.items():
    if manga_id not in non_root_ids:
        # We only keep the fields useful for what follows
        clean_data = {
            'title': manga['title']['romaji'],
            'description': manga['description'],
            'genres': manga['genres'],
            'tags': [tag['name'] for tag in manga['tags']],
            'popularity': manga['popularity']
        }
        clean_root_mangas.append(clean_data)

# 4. Save the clean database
with open('../../data/processed/clean_root_mangas.json', 'w', encoding='utf-8') as f:
    json.dump(clean_root_mangas, f, indent=2, ensure_ascii=False)

print(f"--- Phase 1B (Filtering) Complete! ---")
print(f"Clean database saved: {len(clean_root_mangas)} 'root' mangas.")