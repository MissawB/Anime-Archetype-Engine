import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*50}\n🚀 Exécution de : {script_path}\n{'='*50}")
    # On passe os.environ pour que le subprocess hérite du HF_TOKEN
    result = subprocess.run([sys.executable, script_path], env=os.environ)
    if result.returncode != 0:
        print(f"❌ Erreur lors de l'exécution de {script_path}")
        sys.exit(1)
    print(f"✅ {script_path} terminé avec succès.")

def main():
    # Configuration de l'environnement pour un rendu propre
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
    
    # Charger .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token # Double sécurité
    except ImportError:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_dir = os.path.join(base_dir, 'pipeline')
    
    categories = ['anime', 'manga', 'characters']
    
    for category in categories:
        cat_dir = os.path.join(pipeline_dir, category)
        if not os.path.exists(cat_dir):
            continue
            
        print(f"\n\n{'*'*60}\n🌟 LANCEMENT DE LA PIPELINE : {category.upper()}\n{'*'*60}")
        
        # Les étapes dans l'ordre
        steps = [
            f"1_ingest_{category}.py",
            f"2_refine_{category}.py",
            f"3_filter_{category}.py",
            # "4_train_vibe.py", # Optionnel, on ne le lance pas en auto par défaut
            f"5_vectorize_{category}.py"
        ]
        
        for step in steps:
            script_path = os.path.join(cat_dir, step)
            if os.path.exists(script_path):
                run_script(script_path)
            else:
                print(f"⚠️ Étape ignorée (script introuvable) : {script_path}")

if __name__ == "__main__":
    main()
