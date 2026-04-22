from .utils import TRANSLATIONS

def translation_processor(request):
    # On récupère la langue en session, 'Français' par défaut
    lang = request.session.get('language', 'Français')
    # On renvoie uniquement le dictionnaire correspondant à la langue choisie
    return {
        'txt': TRANSLATIONS.get(lang, TRANSLATIONS['Français']),
        'current_lang': lang
    }
