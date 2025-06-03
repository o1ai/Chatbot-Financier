# TeamWill Finance Assistant

Un chatbot financier RAG (Retrieval-Augmented Generation) optimisé utilisant des modèles Llama/Mistral locaux.

## Fonctionnalités

- **Chatbot financier intelligent** - Répond aux questions sur les produits et services financiers
- **Fonctionne 100% en local** - Utilise des modèles locaux pour la confidentialité et la vitesse
- **Base de connaissances vectorielle** - Utilise ChromaDB avec embeddings BGE pour la recherche sémantique
- **Système anti-hallucination** - Détection et correction automatique des réponses incohérentes
- **Support multilingue** - Français et anglais supportés nativement
- **Filtrage thématique** - Permet de filtrer les réponses par thèmes financiers
- **Auto-optimisation** - S'adapte automatiquement aux ressources système disponibles
- **Métriques et visualisations** - Tableaux de bord pour suivre les performances
- **Interface utilisateur intuitive** - Interface Streamlit moderne et facile à utiliser

## Prérequis

- Python 3.10+ 
- Au moins 4GB de RAM (8GB+ recommandé)
- 5GB+ d'espace disque 

## Installation

1. **Cloner le dépôt**

```bash
git clone [URL_DU_DEPOT]
cd chatbot_teamwill
```

2. **Créer un environnement virtuel (recommandé)**

```bash
python -m venv venv

# Sur Windows
venv\Scripts\activate

# Sur Linux/Mac
source venv/bin/activate
```

3. **Lancer le script d'installation**

```bash
python launch_chatbot.py
```

Ce script va:
- Installer toutes les dépendances nécessaires
- Vérifier que les modèles requis sont présents
- Initialiser la base vectorielle si nécessaire
- Lancer le chatbot avec les paramètres optimaux

## Modèles disponibles

Le système prend en charge plusieurs modèles pré-configurés:

- `Llama-3.2-1B-Instruct-Q5_K_M.gguf` - Modèle léger, recommandé pour les systèmes avec moins de RAM
- `Llama-3.2-3B-Instruct-Q5_K_M.gguf` - Modèle plus performant, recommandé pour de meilleures réponses
- `mistral-7b-instruct-v0.2.Q4_0.gguf` - Modèle avancé, recommandé pour les systèmes avec 12GB+ de RAM

Les modèles peuvent être téléchargés depuis [HuggingFace](https://huggingface.co/TheBloke) et doivent être placés dans le dossier `models/`.

## Utilisation

### Lancement standard

```bash
python launch_chatbot.py
```

### Lancement en mode debug

```bash
python debug_chatbot.py
```

### Ignorer les vérifications de dépendances et modèles

```bash
python launch_chatbot.py --skip-checks
```

## Structure du projet

```
└── chatbot_teamwill/
    ├── chatbot.py            # Application principale du chatbot
    ├── config.py             # Configuration centralisée
    ├── utils.py              # Fonctions utilitaires
    ├── rag_builder.py        # Script pour construire la base vectorielle
    ├── launch_chatbot.py     # Script de lancement optimisé
    ├── debug_chatbot.py      # Version debug du chatbot
    ├── test_carapi.py        # Tests de l'API véhicules
    ├── update_vehicles.py    # Mise à jour des données véhicules
    ├── requirements.txt      # Dépendances du projet
    ├── models/               # Modèles Llama/Mistral
    ├── documents/            # Documents pour la base de connaissances
    ├── chroma_db_bge/        # Base vectorielle ChromaDB
    ├── logs/                 # Logs des interactions
    ├── .cache/              # Cache système
    ├── Ragas/               # Évaluations Ragas
    ├── evaluations/          # Résultats d'évaluation
    └── visualisations/       # Visualisations des performances
```

## Fonctionnalités avancées

### Système anti-hallucination
Le chatbot intègre un système sophistiqué de détection et correction des hallucinations qui :
- Analyse la cohérence des réponses
- Vérifie la correspondance avec les documents sources
- Applique des corrections automatiques si nécessaire

### Support multilingue
- Support natif du français et de l'anglais
- Détection automatique de la langue
- Messages système et réponses adaptés à la langue

### Métriques et visualisations
- Tableaux de bord de performance en temps réel
- Suivi des temps de réponse
- Analyse de la qualité des réponses
- Visualisations des interactions utilisateur

## Optimisation pour la démonstration

Ce chatbot a été optimisé pour:

1. **Performance** - Réponses rapides et chargement optimisé
2. **Robustesse** - Gestion des erreurs et fallback automatique
3. **Qualité de réponse** - Amélioration des prompts et paramètres du modèle
4. **Adaptation automatique** - Configuration adaptée aux ressources disponibles
5. **Fonctionnement 100% local** - Pas de dépendances externes

## Dépannage

### Le modèle est lent à charger
- Utilisez le modèle plus léger `Llama-3.2-1B-Instruct-Q5_K_M.gguf`
- Réduisez le paramètre `n_ctx` dans le fichier `config.py`
- Vérifiez l'utilisation de la mémoire système

### Les réponses sont dans la mauvaise langue
- Vérifiez les paramètres de langue dans l'interface
- Consultez la configuration de langue dans `config.py`
- Vérifiez les documents sources

### Erreurs de mémoire
- Fermez les applications gourmandes en RAM
- Utilisez le modèle plus léger
- Réduisez les paramètres `n_ctx` et `n_batch` dans `config.py`
- Videz le cache du système (`.cache/`)

## Licence

© 2025 TeamWill. Tous droits réservés. 