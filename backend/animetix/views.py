from django.shortcuts import render, redirect
from django.http import JsonResponse
from .services import AnimetixService, LangChainService
import json
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter

# Initialize services
animetix_service = AnimetixService()
try:
    langchain_service = LangChainService()
except Exception as e:
    print(f"Warning: LangChainService could not be initialized: {e}")
    langchain_service = None

DIFFICULTY_SETTINGS = {
    "Easy": {"rank_limit": 500, "power_factor": 3.0},
    "Normal": {"rank_limit": 2000, "power_factor": 2.0},
    "Hard": {"rank_limit": None, "power_factor": 1.0}
}

# --- GLOBAL UTILS ---
def get_current_mode(request):
    return request.session.get('media_type', 'Anime')

def switch_mode(request, mode):
    if mode in ['Anime', 'Manga', 'Character']:
        request.session['media_type'] = mode
        request.session.modified = True
    return redirect(request.META.get('HTTP_REFERER', 'index'))

# --- VIEWS ---

def index(request):
    mode = get_current_mode(request)
    return render(request, 'animetix/index.html', {'current_mode': mode})

def start_game(request):
    media_type = get_current_mode(request)
    difficulty = request.POST.get('difficulty', 'Normal')
    
    request.session['difficulty'] = difficulty
    request.session['media_type'] = media_type
    
    data = animetix_service.load_data(media_type)
    if not data: return redirect('index')
            
    settings_diff = DIFFICULTY_SETTINGS[difficulty]
    rank_limit = settings_diff["rank_limit"]
    
    lookup_data = [item for item in data["lookup"][:rank_limit] if item['title'] in data['title_to_full_data']]
    titles = [item['title'] for item in lookup_data]
    weights_array = np.array([item['popularity'] for item in lookup_data]) + 1.0
    weights = (weights_array ** settings_diff["power_factor"]).tolist()
    
    secret_title = random.choices(titles, weights=weights, k=1)[0]
    
    request.session['secret_title'] = secret_title
    request.session['guesses'] = []
    request.session['game_over'] = False
    request.session['revealed_hints'] = [] # Liste des IDs d'indices révélés
    if 'hint_2' in request.session: del request.session['hint_2']
    
    return redirect('game')

def reveal_hint(request, hint_type):
    revealed = request.session.get('revealed_hints', [])
    if hint_type not in revealed:
        revealed.append(hint_type)
        request.session['revealed_hints'] = revealed
        request.session.modified = True
    return redirect('game')

def game_view(request):
    media_type = request.session.get('media_type')
    if not media_type: return redirect('index')
        
    data = animetix_service.load_data(media_type)
    difficulty = request.session.get('difficulty', 'Normal')
    rank_limit = DIFFICULTY_SETTINGS[difficulty]['rank_limit']
    
    guessed_titles = [g['title'] for g in request.session.get('guesses', [])]
    remaining_items = [item for item in data["lookup"][:rank_limit] if item['title'] not in guessed_titles and item['title'] in data['title_to_full_data']]
    
    secret_title = request.session.get('secret_title')
    secret_data = data['title_to_full_data'].get(secret_title)
    if not secret_data: return redirect('index')

    # Préparation des indices disponibles
    hints = {}
    revealed = request.session.get('revealed_hints', [])
    guess_count = len(guessed_titles)

    # 1. Image (Déblocage immédiat)
    hints['poster'] = {'revealed': 'poster' in revealed, 'value': secret_data.get('image')}
    
    # 2. Recommandation / Similitude (10 essais)
    if media_type == 'Character':
        # Pour les personnages, on montre un personnage "proche" thématiquement
        if 'hint_sim' not in request.session:
            idx_secret = data['title_to_index'][secret_title]
            sims = cosine_similarity(data['vectors_thematic'][idx_secret].reshape(1, -1), data['vectors_thematic'])[0]
            # On prend un personnage très proche (top 5-15 pour ne pas donner la réponse direct)
            top_indices = np.argsort(sims)[::-1]
            hint_char = data['titles'][top_indices[random.randint(5, 15)]]
            request.session['hint_sim'] = hint_char
        hints['sim'] = {'revealed': 'sim' in revealed, 'value': request.session.get('hint_sim'), 'locked': guess_count < 10}
    else:
        if secret_data.get('recommendations'):
            top_rec = max(secret_data['recommendations'], key=secret_data['recommendations'].get)
            hints['rec'] = {'revealed': 'rec' in revealed, 'value': top_rec, 'locked': guess_count < 10}

    # 3. Personnages (pour Anime/Manga) OU Origine (pour Character) (15 essais)
    if media_type != 'Character':
        char_data = animetix_service.load_data('Character')
        if char_data:
            chars = [c['title'] for c in char_data['db'] if c['origin_media'] == secret_title]
            if chars:
                hints['chars'] = {'revealed': 'chars' in revealed, 'value': random.sample(chars, min(2, len(chars))), 'locked': guess_count < 15}
    else:
        # Pour un personnage, l'indice est son œuvre d'origine
        hints['origin'] = {'revealed': 'origin' in revealed, 'value': secret_data.get('origin'), 'locked': guess_count < 15}

    # 4. Mots Rares (20 essais)
    desc = secret_data.get('description', '').lower()
    words = re.findall(r'\w+', desc)
    # Extraction simplifiée de 2 mots de plus de 6 lettres (souvent thématiques)
    rare_candidates = [w for w in words if len(w) > 6 and w not in ['pendant', 'histoire', 'personnages', 'épisodes', 'chapitres']]
    if len(rare_candidates) >= 2:
        hints['words'] = {'revealed': 'words' in revealed, 'value': random.sample(list(set(rare_candidates)), 2), 'locked': guess_count < 20}

    # 5. Indice Vibe IA (25 essais)
    if guess_count >= 25:
        if 'hint_2' not in request.session and langchain_service:
            context_ia = " ".join(secret_data.get('reviews', [])) if media_type != 'Character' else secret_data.get('description', '')
            prompt = f"Donne un indice de 15 mots max sur la 'vibe' ou le rôle du {media_type} '{secret_title}' sans le nommer. Base-toi sur: {context_ia[:1000]}"
            try: 
                res = langchain_service.llm.invoke(prompt)
                request.session['hint_2'] = res.content if hasattr(res, 'content') else str(res)
            except: request.session['hint_2'] = "L'IA est fatiguée."
        hints['vibe'] = {'revealed': 'vibe' in revealed, 'value': request.session.get('hint_2'), 'locked': guess_count < 25}

    context = {
        'media_type': media_type,
        'difficulty': difficulty,
        'guesses': request.session.get('guesses', []),
        'game_over': request.session.get('game_over'),
        'remaining_items': remaining_items,
        'guess_count': guess_count,
        'hints': hints,
        'secret_title': secret_title if request.session.get('game_over') else None,
        'secret_data': secret_data if request.session.get('game_over') else None,
    }

    return render(request, 'animetix/classic/game.html', context)

def abandon_game(request):
    request.session['game_over'] = True
    request.session.modified = True
    return redirect('game')

import requests

def get_similarity_score(mode, secret_idx, guess_idx, data=None):
    """Calcule le score soit localement, soit via le Brain sur Hugging Face."""
    brain_url = os.getenv("BRAIN_API_URL")
    
    if brain_url:
        try:
            # Appel au micro-service sur Hugging Face
            response = requests.post(f"{brain_url}/similarity", json={
                "mode": mode.lower()[:4], # anime, manga, char
                "secret_idx": secret_idx,
                "guess_idx": guess_idx
            }, timeout=2)
            if response.status_code == 200:
                return response.json()["similarity"]
        except Exception as e:
            print(f"Brain API Error: {e}. Falling back to local.")
    
    # Fallback local (si configuré ou si Brain échoue)
    if data and 'vectors_thematic' in data:
        return float(cosine_similarity(
            data['vectors_thematic'][secret_idx].reshape(1, -1), 
            data['vectors_thematic'][guess_idx].reshape(1, -1)
        )[0][0])
    return 0.0

def make_guess(request):
    if request.method == 'POST' and not request.session.get('game_over'):
        guess_title = request.POST.get('guess')
        media_type = get_current_mode(request)
        secret_title = request.session.get('secret_title')
        
        data = animetix_service.load_data(media_type)
        if not guess_title or guess_title not in data['title_to_index']: return redirect('game')
            
        secret_idx = data['title_to_index'][secret_title]
        guess_idx = data['title_to_index'][guess_title]
        secret_full = data['title_to_full_data'][secret_title]
        guess_full = data['title_to_full_data'][guess_title]
        
        # --- CALCUL DE SIMILARITÉ ---
        # On utilise le Brain si disponible, sinon local (Fallback)
        vec_sim = get_similarity_score(media_type, secret_idx, guess_idx, data)

        if media_type == 'Character':
            # Bonus Organisation - 15%
            org_S = set(secret_full.get('organizations', []))
            org_G = set(guess_full.get('organizations', []))
            org_sim = 1.0 if len(org_S.intersection(org_G)) > 0 else 0.0
            
            # Bonus Liens - 15%
            rel_S = set(secret_full.get('related', []))
            rel_G = set(guess_full.get('related', []))
            is_linked = secret_full['title'].lower() in rel_G or guess_full['title'].lower() in rel_S
            link_sim = 1.0 if is_linked else (0.5 if len(rel_S.intersection(rel_G)) > 0 else 0.0)
            
            # Bonus Taille - 10%
            hS, hG = secret_full.get('height_cm', 0), guess_full.get('height_cm', 0)
            height_sim = max(0, 1 - (abs(hS - hG) / 25)) if hS > 0 and hG > 0 else 0
            
            raw_sim = (0.6 * vec_sim) + (0.15 * org_sim) + (0.15 * link_sim) + (0.1 * height_sim)
        else:
            # Logique Anime/Manga avec Recommandations et Plot
            sim_thematic = vec_sim
            sim_plot = float(cosine_similarity(data['vectors_plot'][secret_idx].reshape(1, -1), data['vectors_plot'][guess_idx].reshape(1, -1))[0][0])
            
            rec_rating = 0
            if guess_title in secret_full.get('recommendations', {}):
                rec_rating = secret_full['recommendations'][guess_title]
            elif secret_title in guess_full.get('recommendations', {}):
                rec_rating = guess_full['recommendations'][secret_title]
            
            pop_min = min(secret_full.get('popularity', 1), guess_full.get('popularity', 1))
            sim_rec = min(1.0, np.log1p((rec_rating / pop_min) * 5000) / np.log1p(2)) if rec_rating > 0 else 0

            if rec_rating >= 50:
                raw_sim = (0.65 * sim_thematic + 0.20 * sim_plot + 0.35 * sim_rec) / 1.2
            else:
                raw_sim = (0.80 * sim_thematic + 0.20 * sim_plot)

        # Mise à l'échelle Cémantix
        final_score = round(np.sign(raw_sim) * pow(abs(raw_sim), 4) * 100, 2)

        if final_score > 95: temperature = "BRÛLANT 🔥"; color = "danger"
        elif final_score > 75: temperature = "Chaud ☀️"; color = "warning"
        elif final_score > 50: temperature = "Tiède 🌤️"; color = "primary"
        elif final_score > 25: temperature = "Frais ☁️"; color = "info"
        elif final_score >= 0: temperature = "Glacial ❄️"; color = "secondary"
        else: temperature = "Zéro Absolu 🧊"; color = "dark"

        guesses = request.session.get('guesses', [])
        guesses.append({
            "title": guess_title, "score": final_score, "temp": temperature, "color": color,
            "display_score": max(0, final_score)
        })
        guesses.sort(key=lambda x: x['score'], reverse=True)
        request.session['guesses'] = guesses
        if guess_title == secret_title: request.session['game_over'] = True
        request.session.modified = True
    return redirect('game')

# Mode Archetypist et Paradoxe inchangés...
def archetypist_view(request):
    media_type = get_current_mode(request)
    data = animetix_service.load_data(media_type)
    if request.method == 'POST':
        title_A = request.POST.get('title_A')
        title_B = request.POST.get('title_B')
        if not data: return redirect('index')
        valid_titles = [t for t in data['titles'] if t in data['title_to_full_data']]
        if not title_A or not title_B:
            found = False
            for _ in range(50):
                t1, t2 = random.choices(valid_titles, k=2)
                if t1 != t2:
                    idx1, idx2 = data['title_to_index'][t1], data['title_to_index'][t2]
                    if cosine_similarity(data['vectors_plot'][idx1].reshape(1, -1), data['vectors_plot'][idx2].reshape(1, -1))[0][0] >= 0.70:
                        found = True; break
            if not found: t1, t2 = random.choices(valid_titles, k=2)
        else: t1, t2 = title_A, title_B
        item1, item2 = data['title_to_full_data'][t1], data['title_to_full_data'][t2]
        scenario_data = {"reasoning": "Échec", "scenario": "L'IA est indisponible."}
        if langchain_service:
            scenario_data = langchain_service.generate_scenario_advanced(media_type, item1, item2, "Français")
        return render(request, 'animetix/archetypist/archetypist.html', {
            'media_type': media_type, 'item_A': item1, 'item_B': item2,
            'reasoning': scenario_data.get('reasoning'), 'scenario': scenario_data.get('scenario'), 'show_titles': True
        })
    context = {'items_json': json.dumps(data['lookup']) if data else "[]", 'media_type': media_type}
    return render(request, 'animetix/archetypist/archetypist_form.html', context)

def paradox_view(request):
    media_type = get_current_mode(request)
    data = animetix_service.load_data(media_type)
    if request.method == 'POST':
        if not data: return redirect('index')
        valid_titles = [t for t in data['titles'] if t in data['title_to_full_data']]
        t1, t2 = random.choices(valid_titles[:500], k=2)
        idx1 = data['title_to_index'][t1]
        intruder = None
        for _ in range(30):
            ti = random.choice(valid_titles)
            if ti not in [t1, t2]:
                idxi = data['title_to_index'][ti]
                if cosine_similarity(data['vectors_thematic'][idx1].reshape(1, -1), data['vectors_thematic'][idxi].reshape(1, -1))[0][0] < 0.40:
                    intruder = ti; break
        if not intruder: intruder = random.choice(valid_titles)
        item1, item2 = data['title_to_full_data'][t1], data['title_to_full_data'][t2]
        scenario_data = {"scenario": "Erreur génération."}
        if langchain_service:
            scenario_data = langchain_service.generate_scenario_advanced(media_type, item1, item2, "Français")
        options = [t1, t2, intruder]
        random.shuffle(options)
        request.session['paradox_answer'] = intruder
        request.session['paradox_media'] = media_type
        return render(request, 'animetix/paradox/intruder.html', {
            'scenario': scenario_data.get('scenario'), 'options': options, 'media_type': media_type, 'reasoning': scenario_data.get('reasoning')
        })
    return render(request, 'animetix/paradox/paradox_form.html', {'media_type': media_type})

def paradox_guess(request):
    if request.method == 'POST':
        user_choice = request.POST.get('choice')
        answer = request.session.get('paradox_answer')
        media_type = request.session.get('paradox_media')
        data = animetix_service.load_data(media_type)
        return render(request, 'animetix/paradox/intruder_result.html', {
            'is_correct': (user_choice == answer), 'answer': answer,
            'answer_data': data['title_to_full_data'][answer], 'media_type': media_type
        })
    return redirect('index')

def undercover_party_setup(request):
    return render(request, 'animetix/undercover/undercover_setup.html', {'media_type': get_current_mode(request)})

def undercover_party_play(request):
    if request.method == 'POST':
        media_type = get_current_mode(request)
        num_players = int(request.POST.get('num_players', 3))
        difficulty = request.POST.get('difficulty', 'Normal')
        data = animetix_service.load_data(media_type)
        valid_titles = [t for t in data['titles'] if t in data['title_to_full_data']]
        
        # Sélection du Civil (parmi les populaires pour éviter les inconnus)
        civil_title = random.choice(valid_titles[:300])
        idx_civil = data['title_to_index'][civil_title]
        
        # Similarité thématique pour trouver l'Undercover
        similarities = cosine_similarity(data['vectors_thematic'][idx_civil].reshape(1, -1), data['vectors_thematic'])[0]
        thresholds = {"Easy": (0.35, 0.60), "Normal": (0.65, 0.85), "Hard": (0.86, 0.98)}
        low, high = thresholds.get(difficulty, thresholds["Normal"])
        
        candidates = [i for i, s in enumerate(similarities) if low <= s <= high and data['titles'][i] != civil_title]
        undercover_title = data['titles'][random.choice(candidates)] if candidates else random.choice(valid_titles)
        
        civil_obj = data['title_to_full_data'][civil_title]
        undercover_obj = data['title_to_full_data'][undercover_title]
        
        clue = "Le secret est bien gardé..."
        icon = ""
        if langchain_service:
            icon = langchain_service._get_icon()
            clue = langchain_service.generate_undercover_clue(media_type, civil_title, undercover_title, "Français")
            
        players = []
        undercover_index = random.randint(0, num_players - 1)
        for i in range(num_players):
            role = "Undercover" if i == undercover_index else "Civil"
            obj = undercover_obj if role == "Undercover" else civil_obj
            players.append({
                "id": i+1, "role": role, "title": obj['title'], "title_en": obj.get('title_english'),
                "title_nat": obj.get('title_native'), "image": obj.get('image'),
            })
        return render(request, 'animetix/undercover/undercover_party.html', {'num_players': num_players, 'players': players, 'clue': clue, 'icon': icon})
    return redirect('index')
