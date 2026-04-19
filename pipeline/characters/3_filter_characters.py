import json
import os

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_file = os.path.join(BASE_DIR, 'data', 'processed', 'refined_characters.json')
    output_file = os.path.join(BASE_DIR, 'data', 'processed', 'filtered_characters.json')

    if not os.path.exists(input_file):
        print("⚠️ Fichier raffiné non trouvé. Lancez d'abord 2_refine_characters.py")
        return

    print("Filtrage de la base personnages...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Exemple de filtrage : on ne garde que les personnages avec une image et une biographie décente
    filtered_chars = []
    for char in data:
        # On peut ajouter des critères ici (ex: popularité minimale, etc.)
        if char.get('image') and len(char.get('biography', '')) > 20:
            filtered_chars.append(char)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_chars, f, indent=2, ensure_ascii=False)

    print(f"✅ {len(filtered_chars)} personnages sélectionnés (sur {len(data)}).")

if __name__ == "__main__":
    main()
