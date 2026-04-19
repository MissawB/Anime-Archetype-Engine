# 🧩 Animetix : IA & Game-Design pour l'exploration Otaku

**Animetix** est une plateforme interactive qui réinvente la découverte d'Animes, Mangas et Personnages grâce à l'Intelligence Artificielle. Le projet utilise une architecture hybride "Brain-Face" pour offrir des calculs de similarité sémantique en temps réel sans compromis sur la performance.

## 🚀 Architecture Hybride
- **Le Visage (Frontend/Web)** : Propulsé par Django et hébergé sur **Koyeb**. Gère l'expérience utilisateur, les sessions de jeu et l'interface fluide.
- **Le Cerveau (API IA)** : Un micro-service FastAPI hébergé sur **Hugging Face Spaces** ([MissawB/animetix-brain](https://huggingface.co/spaces/MissawB/animetix-brain)). Il gère 16 Go de vecteurs d'embeddings pour calculer instantanément la "distance culturelle" entre deux entités.

## 🎮 Modes de Jeu
- **Animetix Classic** : Un "Cémantix" version Otaku. Trouvez l'œuvre secrète via des scores de similarité thématique et de scénario.
- **Story Archetypist** : Fusion de concepts via LLM (Llama 3.2 / Gemini). L'IA crée des synopsis originaux mélangeant les univers.
- **Undercover Party** : Jeu de déduction sociale local où l'IA génère des indices subtils pour débusquer l'intrus.

## 🛠️ Stack Technique
- **Backend** : Python 3.11, Django 6.0, FastAPI.
- **IA/NLP** : Sentence-Transformers (`all-mpnet-base-v2`), Scikit-learn, Hugging Face Hub.
- **Orchestration** : Docker, Docker Compose, ETL Pipeline personnalisée.

## 🚀 Installation & Lancement

1. **Prérequis** : `pip install -r requirements.txt`
2. **Clés API** : Créer un `.env` avec `GEMINI_API_KEY` et `BRAIN_API_URL`.
3. **Data** : `python run_pipeline.py` pour générer les vecteurs.
4. **Web** : `cd backend && python manage.py runserver`

---
Hébergé avec ❤️ par [MissawB](https://huggingface.co/MissawB)
