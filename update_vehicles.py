"""
Script de mise à jour automatique de la base de données des véhicules.
"""

import schedule
import time
from utils import update_vehicle_database
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vehicle_updates.log'),
        logging.StreamHandler()
    ]
)

def main():
    """
    Fonction principale qui gère la mise à jour de la base de données.
    """
    try:
        # Créer le dossier de logs s'il n'existe pas
        Path('logs').mkdir(exist_ok=True)
        
        logging.info("Démarrage du service de mise à jour des véhicules")
        
        # Première mise à jour au démarrage
        update_vehicle_database()
        
        # Planifier les mises à jour quotidiennes
        schedule.every().day.at("02:00").do(update_vehicle_database)  # Mise à jour à 2h du matin
        
        # Boucle principale
        while True:
            schedule.run_pending()
            time.sleep(60)  # Attendre 1 minute avant la prochaine vérification
            
    except Exception as e:
        logging.error(f"Erreur dans le service de mise à jour : {str(e)}")
        raise

if __name__ == "__main__":
    main() 