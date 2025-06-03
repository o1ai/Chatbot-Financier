"""
Configuration pour l'évaluation RAGAS
Personnalisez ici les paramètres d'évaluation selon vos besoins
"""

from pathlib import Path

# === CONFIGURATION D'ÉVALUATION ===

# Stratégie de sélection des interactions
EVALUATION_STRATEGY = {
    # Mode: "latest_day", "all_days", "sample", "last_n_days"
    "mode": "all_days",
    
    # Nombre maximum d'interactions à évaluer (None = toutes)
    "max_samples": None,
    
    # Nombre de jours à inclure (pour mode "last_n_days")
    "n_days": 7,
    
    # Fichiers spécifiques (pour mode "custom_files")
    "custom_files": []
}

# Métriques à évaluer
RAGAS_METRICS = {
    "faithfulness": True,           # Fidélité aux sources
    "answer_relevancy": True,       # Pertinence des réponses  
    "context_precision": True,      # Précision du contexte
    "context_recall": True          # Rappel du contexte
}

# Configuration des seuils d'alerte
PERFORMANCE_THRESHOLDS = {
    "excellent": 0.8,       # Score excellent (vert)
    "good": 0.6,           # Score correct (jaune)
    "poor": 0.4,           # Score faible (rouge)
}

# Filtres d'exclusion
EXCLUSION_FILTERS = {
    # Exclure les questions trop courtes
    "min_question_length": 5,
    
    # Exclure les réponses d'erreur
    "exclude_error_responses": True,
    
    # Exclure certains thèmes
    "excluded_themes": [],
    
    # Exclure certaines langues
    "excluded_languages": []
}

# Configuration des rapports
REPORT_CONFIG = {
    # Générer des graphiques
    "generate_charts": True,
    
    # Envoyer des notifications
    "send_notifications": False,
    
    # Email pour notifications
    "notification_email": None,
    
    # Seuil pour déclencher une alerte
    "alert_threshold": 0.5
}

# === FONCTIONS UTILITAIRES ===

def get_evaluation_mode():
    """Retourne le mode d'évaluation configuré"""
    return EVALUATION_STRATEGY["mode"]

def should_include_interaction(interaction):
    """Détermine si une interaction doit être incluse dans l'évaluation"""
    
    # Vérifier la longueur de la question
    if len(interaction.get("question", "")) < EXCLUSION_FILTERS["min_question_length"]:
        return False
    
    # Exclure les réponses d'erreur
    if EXCLUSION_FILTERS["exclude_error_responses"]:
        answer = interaction.get("answer", "").lower()
        error_keywords = ["erreur", "error", "une erreur s'est produite", "impossible"]
        if any(keyword in answer for keyword in error_keywords):
            return False
    
    # Vérifier la langue
    detected_lang = interaction.get("metadata", {}).get("detected_language")
    if detected_lang in EXCLUSION_FILTERS["excluded_languages"]:
        return False
    
    return True

def get_performance_level(score):
    """Retourne le niveau de performance basé sur le score"""
    if score >= PERFORMANCE_THRESHOLDS["excellent"]:
        return "excellent", "🟢"
    elif score >= PERFORMANCE_THRESHOLDS["good"]:
        return "good", "🟡" 
    elif score >= PERFORMANCE_THRESHOLDS["poor"]:
        return "poor", "🟠"
    else:
        return "critical", "🔴"

# === EXEMPLES DE CONFIGURATION ===

def config_daily_evaluation():
    """Configuration pour évaluation quotidienne (fichier du jour seulement)"""
    EVALUATION_STRATEGY["mode"] = "latest_day"
    EVALUATION_STRATEGY["max_samples"] = None

def config_sample_evaluation(sample_size=50):
    """Configuration pour évaluation sur échantillon"""
    EVALUATION_STRATEGY["mode"] = "sample"
    EVALUATION_STRATEGY["max_samples"] = sample_size

def config_full_evaluation():
    """Configuration pour évaluation complète (toutes les interactions)"""
    EVALUATION_STRATEGY["mode"] = "all_days"
    EVALUATION_STRATEGY["max_samples"] = None

def config_weekly_evaluation():
    """Configuration pour évaluation hebdomadaire"""
    EVALUATION_STRATEGY["mode"] = "last_n_days"
    EVALUATION_STRATEGY["n_days"] = 7
    EVALUATION_STRATEGY["max_samples"] = None

# Appliquer la configuration par défaut
config_full_evaluation()  # Évaluer TOUTES les interactions par défaut 