"""
Script de test pour l'intégration CARAPI
"""

import os
from utils import get_carapi_jwt, get_vehicle_data_from_api
from dotenv import load_dotenv

def test_auth():
    """Teste l'authentification CARAPI"""
    print("\n=== Test d'authentification ===")
    jwt = get_carapi_jwt()
    if jwt:
        print("✅ JWT obtenu avec succès")
        print(f"Longueur du token: {len(jwt)} caractères")
    else:
        print("❌ Échec de l'obtention du JWT")

def test_vehicle_search():
    """Teste la recherche de véhicules"""
    print("\n=== Test de recherche de véhicules ===")
    
    # Test sans filtre pour voir toutes les données disponibles
    print("Recherche sans filtre...")
    data = get_vehicle_data_from_api()
    
    if data:
        print("✅ Données reçues avec succès")
        print(f"Catégories disponibles: {list(data['categories'].keys())}")
        print("\nExemple de véhicules par catégorie:")
        for category, vehicles in data["vehicules"].items():
            print(f"\n{category.upper()}:")
            for vehicle in vehicles[:2]:  # Afficher les 2 premiers véhicules de chaque catégorie
                print(f"- {vehicle['marque']} {vehicle['modele']} ({vehicle['version']})")
                print(f"  Prix: {vehicle['prix']}€")
                print("  Options de financement disponibles:")
                for option in vehicle['options_financement'].keys():
                    print(f"  - {option.upper()}")
    else:
        print("❌ Échec de la récupération des données")

def main():
    """Fonction principale de test"""
    # Charger les variables d'environnement
    load_dotenv()
    
    # Vérifier les credentials
    print("=== Vérification des credentials ===")
    token = os.getenv("CARAPI_TOKEN")
    secret = os.getenv("CARAPI_SECRET")
    
    print(f"Token présent: {'✅' if token else '❌'}")
    print(f"Secret présent: {'✅' if secret else '❌'}")
    
    # Exécuter les tests
    test_auth()
    test_vehicle_search()

if __name__ == "__main__":
    main() 