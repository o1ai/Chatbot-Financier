"""
Configuration centralisée pour le chatbot de financement automobile.
Ce fichier contient les paramètres partagés entre toutes les interfaces et composants.
"""

import os
from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
DOCUMENTS_DIR = BASE_DIR / "documents"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / ".cache"

# Création des répertoires nécessaires
for directory in [MODELS_DIR, DOCUMENTS_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# Limites et seuils
MAX_INPUT_LENGTH = 1000
MAX_CONTEXT_LENGTH = 4000
MIN_RESPONSE_LENGTH = 50
SEARCH_TOP_K = 5
SIMILARITY_THRESHOLD = 0.75
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Configuration de la base vectorielle optimisée
VECTORDB_CONFIG = {
    'persist_directory': str(BASE_DIR / "chroma_db_bge"),
    'embedding_model': "BAAI/bge-small-fr",
    'model_kwargs': {'device': 'cpu'},
    'chunk_size': 300,          # Réduit pour plus de précision
    'chunk_overlap': 100,       # Augmenté pour meilleure cohérence
    'distance_metric': 'cosine',
    'fetch_k': 10              # Augmenté pour plus de diversité
}

# Configuration des modèles par taille de mémoire
MODEL_CONFIGS = {
    'small': {
        'name': "Llama-3.2-1B-Instruct-Q5_K_M.gguf",
        'min_memory': 4,
        'n_ctx': 1024,
        'n_batch': 256
    },
    'medium': {
        'name': "Llama-3.2-3B-Instruct-Q5_K_M.gguf",
        'min_memory': 8,
        'n_ctx': 2048,
        'n_batch': 512
    },
    'large': {
        'name': "mistral-7b-instruct-v0.2.Q4_0.gguf",
        'min_memory': 12,
        'n_ctx': 4096,
        'n_batch': 1024
    }
}

# Configuration optimisée du modèle
MODEL_CONFIG = {
    'model_path': str(MODELS_DIR / MODEL_CONFIGS['small']['name']),
    'n_ctx': 2048,          # Contexte réduit pour plus de rapidité
    'n_batch': 512,         # Taille de batch optimisée
    'n_threads': 4,         # Nombre de threads optimal
    'temperature': 0.3,     # Température réduite pour des réponses plus précises
    'top_p': 0.85,         # Filtrage plus strict des tokens
    'repeat_penalty': 1.2,  # Pénalité accrue pour les répétitions
    'max_tokens': 150,      # Limite de longueur pour des réponses concises
    'verbose': False
}

# Messages d'erreur multilingues
LANGUAGE_MESSAGES = {
    "fr": {
        "greeting": "Bonjour ! Je suis votre assistant financier TeamWill. Comment puis-je vous aider aujourd'hui ?",
        "no_info": "Je n'ai pas cette information dans ma base de connaissances.",
        "error": "Une erreur s'est produite. Pouvez-vous reformuler votre question ?",
        "fallback": "Je n'ai pas pu générer une réponse appropriée. Pouvez-vous reformuler votre question ?",
        "too_long": "Votre message est trop long. Pouvez-vous le raccourcir ?",
        "timeout": "Le temps de réponse a été dépassé. Pouvez-vous reformuler votre question ?"
    },
    "en": {
        "greeting": "Hello! I'm your TeamWill financial assistant. How can I help you today?",
        "no_info": "I don't have this information in my knowledge base.",
        "error": "An error occurred. Could you please rephrase your question?",
        "fallback": "I couldn't generate an appropriate response. Could you please rephrase your question?",
        "too_long": "Your message is too long. Could you make it shorter?",
        "timeout": "The response time was exceeded. Could you rephrase your question?"
    }
}

# Thèmes disponibles
AVAILABLE_THEMES = {
    "all": "Tous les documents",
    "products": "Produits financiers",
    "services": "Services bancaires",
    "investments": "Investissements",
    "insurance": "Assurances",
    "regulations": "Réglementation"
}

# Configuration du logging
LOG_CONFIG = {
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'level': 'INFO',
    'rotation': '1 day',
    'retention': '7 days',
    'compression': 'zip'
}

# Configuration de la surveillance système
MONITORING_CONFIG = {
    'memory_warning': 80,  # Pourcentage
    'cpu_warning': 85,     # Pourcentage
    'disk_warning': 90,    # Pourcentage
    'check_interval': 60   # Secondes
}

# Configuration du cache
CACHE_CONFIG = {
    'ttl': 1800,           # 30 minutes de cache
    'max_size': '2GB',     # Taille de cache augmentée
    'cleanup_interval': 600 # Nettoyage toutes les 10 minutes
}

# Configuration des timeouts
TIMEOUT_CONFIG = {
    'model_load': 30,      # Réduit à 30 secondes
    'query': 15,           # Réduit à 15 secondes
    'api_call': 5         # Réduit à 5 secondes
}

# Configuration des retry
RETRY_CONFIG = {
    'max_attempts': 3,
    'backoff_factor': 2,
    'max_delay': 10
}

# Paramètres de performance
PERFORMANCE_CONFIG = {
    'batch_size': 32,
    'num_workers': 4,
    'prefetch_factor': 2,
    'pin_memory': True
}

# Configuration de l'API
API_CONFIG = {
    'base_url': 'https://api.teamwill.fr',
    'timeout': 10,
    'max_retries': 3,
    'verify_ssl': True
}

# Configuration des prompts
PROMPT_CONFIG = {
    'max_length': 2048,
    'temperature': 0.7,
    'top_p': 0.95,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}

# Paramètres d'évaluation
EVAL_CONFIG = {
    'min_confidence': 0.8,
    'max_tokens': 100,
    'metrics': ['relevance', 'coherence', 'fluency']
}

# Configuration du text_splitter
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 500,        # Chunks plus petits pour des recherches plus précises
    "chunk_overlap": 100,     # Chevauchement réduit pour améliorer les performances
    "separators": ["\n\n", "\n", ". ", " ", ""],
    "keep_separator": True,
    "length_function": len
}

# Configuration du retriever optimisée
RETRIEVER_CONFIG = {
    "k": 3,                    # Nombre final de documents
    "fetch_k": 10,             # Documents initiaux à considérer
    "lambda_mult": 0.6,        # Favorise la pertinence vs diversité
    "use_compression": True,   # Active la compression contextuelle
    "compression_ratio": 0.7   # Garde 70% du contenu après compression
}

# Thèmes disponibles pour le filtrage avec descriptions
THEMES = [
    "Tous",
    "financement_auto",
    "credit_auto",
    "leasing",
    "loa",
    "lld",
    "assurance",
    "profil_client"
]

# Descriptions des thèmes pour l'interface utilisateur
THEME_DESCRIPTIONS = {
    "Tous": "Toutes les solutions de financement",
    "financement_auto": "Vue d'ensemble des solutions de financement automobile",
    "credit_auto": "Crédit automobile classique",
    "leasing": "Solutions de leasing automobile",
    "loa": "Location avec Option d'Achat",
    "lld": "Location Longue Durée",
    "assurance": "Assurances et garanties",
    "profil_client": "Analyse et recommandations selon profil"
}

# Critères de profilage client
PROFIL_CRITERIA = {
    "budget_mensuel": {
        "faible": "< 300€",
        "moyen": "300€ - 600€",
        "élevé": "> 600€"
    },
    "usage": {
        "personnel": "Usage personnel/familial",
        "professionnel": "Usage professionnel",
        "mixte": "Usage mixte"
    },
    "duree": {
        "court": "1-2 ans",
        "moyen": "3-4 ans",
        "long": "5 ans et plus"
    },
    "situation": {
        "salarie": "Salarié en CDI",
        "independant": "Travailleur indépendant",
        "entreprise": "Entreprise",
        "autre": "Autre situation"
    }
}

# Mots-clés pour le filtrage thématique
THEME_KEYWORDS = {
    "financement_auto": [
        "financement", "mensualité", "budget", "apport", "durée",
        "taux", "simulation", "comparaison"
    ],
    "credit_auto": [
        "crédit", "prêt", "taux fixe", "mensualité", "apport",
        "propriété", "remboursement", "amortissement"
    ],
    "leasing": [
        "leasing", "location", "option achat", "loyer", "durée",
        "kilométrage", "entretien", "services"
    ],
    "loa": [
        "LOA", "option achat", "premier loyer", "apport", "valeur résiduelle",
        "rachat", "fin contrat", "engagement"
    ],
    "lld": [
        "LLD", "location longue durée", "loyer mensuel", "entretien",
        "services inclus", "kilométrage", "restitution"
    ],
    "assurance": [
        "assurance", "garantie", "couverture", "franchise", "sinistre",
        "assistance", "protection", "responsabilité"
    ],
    "profil_client": [
        "profil", "situation", "revenus", "usage", "besoins",
        "préférences", "contraintes", "objectifs"
    ]
}

# Règles de recommandation
RECOMMENDATION_RULES = {
    "credit_auto": {
        "ideal_pour": [
            "Clients souhaitant devenir propriétaire",
            "Budget mensuel stable",
            "Usage long terme",
            "Possibilité d'apport initial"
        ],
        "avantages": [
            "Propriété du véhicule",
            "Pas de contrainte kilométrique",
            "Possibilité de revente",
            "Taux fixe"
        ]
    },
    "loa": {
        "ideal_pour": [
            "Clients souhaitant changer régulièrement de véhicule",
            "Budget mensuel maîtrisé",
            "Usage mixte",
            "Préférence pour véhicules neufs"
        ],
        "avantages": [
            "Mensualités réduites",
            "Option d'achat en fin de contrat",
            "Services inclus possibles",
            "Renouvellement facilité"
        ]
    },
    "lld": {
        "ideal_pour": [
            "Professionnels",
            "Entreprises",
            "Usage intensif",
            "Besoin de services inclus"
        ],
        "avantages": [
            "Pas d'immobilisation financière",
            "Services tout compris",
            "Fiscalité avantageuse",
            "Gestion simplifiée"
        ]
    }
}

# Prompt système renforcé
SYSTEM_PROMPT = """Tu es un assistant spécialisé en financement automobile pour TeamWill.

INSTRUCTIONS STRICTES ET NON NÉGOCIABLES:
1. Utilise EXCLUSIVEMENT les informations du contexte fourni
2. Si une information n'est pas dans le contexte, réponds UNIQUEMENT: "Je n'ai pas cette information dans mes données actuelles."
3. Ne fais JAMAIS de suppositions ou de généralisations
4. Limite TOUJOURS tes réponses à 2 phrases maximum
5. N'ajoute AUCUN commentaire personnel
6. N'utilise JAMAIS de termes vagues ou incertains
7. Ne donne JAMAIS d'informations non vérifiées

FORMAT DE RÉPONSE:
- Sois direct et précis
- Pas d'introduction ni de conclusion
- Uniquement les faits du contexte"""

# Version de l'application
VERSION = "1.0.0"

# Configuration de la validation des réponses
VALIDATION_CONFIG = {
    'min_confidence': 0.7,     # Seuil minimum de confiance
    'max_length': 100,         # Longueur maximale en mots
    'min_context_overlap': 3,  # Minimum de mots-clés du contexte
    'banned_phrases': [        # Phrases à rejeter
        "je pense que",
        "peut-être",
        "probablement",
        "il me semble",
        "il est possible",
        "généralement",
        "habituellement",
        "en général",
        "parfois",
        "souvent"
    ]
} 