# Image de base stable et légère
FROM python:3.11-slim

# Éviter la génération de fichiers .pyc et forcer l'affichage des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système nécessaires pour certaines libs ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Création des dossiers de données s'ils n'existent pas
RUN mkdir -p data/raw data/processed data/models data/artifacts

# Port par défaut pour Django
EXPOSE 8000

# Commande par défaut (sera surchargée par docker-compose)
CMD ["python", "backend/manage.py", "runserver", "0.0.0.0:8000"]
