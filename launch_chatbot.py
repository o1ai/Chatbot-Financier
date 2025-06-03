#!/usr/bin/env python
"""
Script de lancement optimisé pour le chatbot TeamWill Finance.
Ce script prépare l'environnement et lance le chatbot avec des paramètres optimisés.
"""

import os
import sys
import subprocess
import argparse
import platform
from pathlib import Path

# Définir le répertoire de base
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    try:
        print("Vérification des dépendances...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", BASE_DIR / "requirements.txt"])
        return True
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation des dépendances.")
        return False

def check_models():
    """Vérifie que les modèles nécessaires sont présents."""
    models_dir = BASE_DIR / "models"
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
    
    required_model = models_dir / "Llama-3.2-1B-Instruct-Q5_K_M.gguf"
    
    if not required_model.exists():
        print(f"❌ Modèle requis non trouvé: {required_model.name}")
        print("Veuillez télécharger le modèle depuis https://huggingface.co/TheBloke/Llama-3.2-1B-Instruct-GGUF")
        print(f"et le placer dans le répertoire {models_dir}")
        return False
    
    return True

def check_vectordb():
    """Vérifie que la base vectorielle est initialisée."""
    vectordb_dir = BASE_DIR / "chroma_db_bge"
    
    if not vectordb_dir.exists() or not any(vectordb_dir.iterdir()):
        print("⚠️ Base vectorielle non initialisée.")
        print("Exécution du script de création de la base...")
        try:
            subprocess.check_call([sys.executable, BASE_DIR / "rag_builder.py"])
            print("✅ Base vectorielle initialisée avec succès.")
        except subprocess.CalledProcessError:
            print("❌ Erreur lors de l'initialisation de la base vectorielle.")
            return False
    
    return True

def set_environment():
    """Configure l'environnement pour des performances optimales."""
    os_name = platform.system()
    
    # Variables d'environnement communes
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Évite les avertissements
    os.environ["PYTHONHASHSEED"] = "42"  # Pour la reproductibilité
    
    # Variables spécifiques à Windows
    if os_name == "Windows":
        # Configuration pour Windows
        os.environ["OMP_NUM_THREADS"] = "4"  # Limiter le nombre de threads OpenMP
        os.environ["MKL_NUM_THREADS"] = "4"  # Limiter le nombre de threads MKL
    
    # Variables spécifiques à Linux/macOS
    else:
        # Configuration pour Linux/macOS
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["OPENBLAS_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
        os.environ["NUMEXPR_NUM_THREADS"] = "4"

def launch_chatbot(debug=False):
    """Lance le chatbot avec les paramètres optimaux."""
    streamlit_cmd = [sys.executable, "-m", "streamlit", "run", BASE_DIR / "chatbot.py"]
    
    # Options Streamlit pour améliorer les performances
    streamlit_options = [
        "--server.maxUploadSize=10",  # Limiter la taille des téléchargements à 10 Mo
        "--server.enableXsrfProtection=false",  # Désactiver la protection XSRF pour des performances
        "--browser.serverAddress=localhost",  # Forcer l'adresse localhost
        "--theme.base=light",  # Thème clair
    ]
    
    if debug:
        # Mode debug avec logs détaillés
        streamlit_options.append("--logger.level=debug")
        streamlit_options.append("--server.enableCORS=false")
    else:
        # Mode production avec moins de logs
        streamlit_options.append("--logger.level=warning")
        streamlit_options.append("--server.headless=true")  # Mode sans tête (pas de navigateur auto)
    
    print("🚀 Lancement du chatbot TeamWill Finance...")
    subprocess.run(streamlit_cmd + streamlit_options)

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Lance le chatbot TeamWill Finance avec des paramètres optimisés.")
    parser.add_argument("--debug", action="store_true", help="Lance le chatbot en mode debug")
    parser.add_argument("--skip-checks", action="store_true", help="Ignore les vérifications de dépendances et modèles")
    args = parser.parse_args()
    
    print("=== TeamWill Finance Assistant ===")
    
    if not args.skip_checks:
        # Vérifier les dépendances et modèles
        if not check_dependencies():
            sys.exit(1)
        
        if not check_models():
            sys.exit(1)
        
        if not check_vectordb():
            print("⚠️ Continuer malgré l'erreur de base vectorielle? (y/n)")
            if input().lower() != 'y':
                sys.exit(1)
    
    # Configurer l'environnement
    set_environment()
    
    # Lancer le chatbot
    launch_chatbot(debug=args.debug)

if __name__ == "__main__":
    main() 