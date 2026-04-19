# 🧩 Animetix : IA & Culture Otaku

**Animetix** est une plateforme web interactive propulsée par **Django** et **LangChain**, conçue pour explorer les univers d'animés et de mangas à travers des algorithmes de similarité sémantique et des modèles de langage de pointe (**Llama 3.2 3B** & **Gemini 2.5**).

---

## 🎮 Modes de Jeu

### 🎯 Animetix Classic
Trouvez l'œuvre secrète ! Chaque proposition reçoit un score de similarité basé sur un modèle hybride :
*   **80% Thématique** : Tags et genres (poids fort).
*   **20% Plot** : Structure du scénario.

### 🎬 Story Archetypist (Fusion)
Choisissez deux œuvres et laissez l'IA créer un **synopsis original** fusionnant leurs concepts (systèmes de magie, enjeux sociaux, atmosphère). L'IA est bridée pour ne citer aucun nom d'œuvre ou de personnage.

### 🧩 Le Paradoxe d'Archetyp
L'IA fusionne deux œuvres secrètes. Saurez-vous identifier lequel des 3 titres proposés est l'**intrus** qui n'a servi à aucune fusion ?

### 🕵️ Undercover Party
Un jeu de déduction sociale pour 3 à 12 joueurs (local/hotseat) :
*   Tout le monde reçoit la même œuvre, sauf un **Undercover** qui en reçoit une proche thématiquement.
*   L'IA génère un indice commun secret pour lancer le débat.
*   Interface interactive pour éliminer les suspects et révéler les rôles.

---

## 🛠️ Architecture & IA

### Structure Professionnelle du Projet
L'architecture a été séparée pour maximiser la modularité entre l'application web et la chaîne de traitement des données (ETL).

*   `backend/` : Le cœur de l'application web Django.
    *   `manage.py` : Script de gestion Django.
    *   `animetix/` : Application principale avec les vues et la logique métier (`services.py`).
    *   `animetix_project/` : Configuration globale du projet.
*   `pipeline/` : Pipeline ETL (Extract, Transform, Load) pour le traitement NLP et ML.
    *   `anime/`, `manga/`, `characters/` : Scripts de collecte, raffinage, filtrage, fine-tuning et vectorisation.
*   `data/` : Espace de stockage des données.
    *   `raw/` : Données brutes issues de l'API AniList.
    *   `processed/` : Données nettoyées et raffinées par l'IA.
    *   `models/` : Modèles de Sentence-Transformers fine-tunés.
    *   `artifacts/` : Embeddings numpy (`.npy`) et dictionnaires de recherche utilisés par le backend.
*   `run_pipeline.py` : Script d'orchestration permettant de lancer l'intégralité de la pipeline de données d'un seul coup.

### Moteurs IA
*   **Llama 3.2 3B** (Local via Ollama 🦙) : Utilisé en priorité pour sa rapidité et sa gratuité.
*   **Gemini 2.5 Flash** (Cloud via Google API ✨) : Système de repli automatique (Fallback) garantissant la continuité du service.

---

## 🚀 Installation & Lancement

1.  **Prérequis** :
    ```bash
    pip install -r requirements.txt
    ```
2.  **Ollama (Optionnel)** : Installez [Ollama](https://ollama.com/) et lancez `ollama run llama3.2:3b`.
3.  **Clés API** : Créez un fichier `.env` à la racine avec votre `GEMINI_API_KEY`.
4.  **Génération des données (ETL)** :
    ```bash
    python run_pipeline.py
    ```
5.  **Lancement du Serveur Web** :
    ```bash
    cd backend
    python manage.py runserver
    ```

---

## 🧪 Technologies
*   **Backend** : Django 6.0
*   **NLP** : Sentence-Transformers (`all-mpnet-base-v2`), Scikit-learn (Cosine Similarity).
*   **Frontend** : Bootstrap 5, Tom Select (Recherche intelligente).
