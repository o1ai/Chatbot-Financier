"""
Chatbot TeamWill - Interface unifiée
Application Streamlit pour interagir avec le chatbot RAG basé sur Mistral.
"""

import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os
import tempfile
import datetime as dt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import re
import json
import psutil

import config
import utils
from utils.hallucination_detection import HallucinationDetector

# Initialiser le détecteur d'hallucinations
hallucination_detector = HallucinationDetector()

# === Fonction de post-traitement anti-hallucination ===
def post_process_answer(answer, source_documents, detected_lang, question=None):
    """
    Post-traite la réponse avec le nouveau système de détection d'hallucinations.
    
    Args:
        answer: La réponse du modèle
        source_documents: Les documents sources
        detected_lang: Langue détectée
        question: Question originale (optionnelle)
    
    Returns:
        Réponse corrigée
    """
    if not answer or not answer.strip():
        return utils.get_language_message(detected_lang, "fallback")
    
    try:
        # Préparer le contexte à partir des documents sources
        context = " ".join([doc.page_content for doc in source_documents]) if source_documents else ""
        
        # Déterminer le type de question
        question_type = 'général'
        if question:
            question_lower = question.lower()
            if any(kw in question_lower for kw in ['combien', 'quand', 'où', 'qui']):
                question_type = 'factuel'
            elif any(kw in question_lower for kw in ['penses-tu', 'crois-tu', 'opinion']):
                question_type = 'opinion'
            elif any(kw in question_lower for kw in ['comment', 'explique', 'décris']):
                question_type = 'technique'
        
        # Valider la réponse
        validation_result = hallucination_detector.validate_response(
            answer=answer.strip(),
            context=context,
            question=question if question else "",
            question_type=question_type
        )
        
        # Logger le résultat pour analyse
        hallucination_detector.log_validation_result(
            result=validation_result,
            metadata={
                'timestamp': dt.datetime.now().isoformat(),
                'detected_lang': detected_lang,
                'question_type': question_type
            }
        )
        
        # Si la réponse n'est pas valide, utiliser la correction suggérée ou le fallback
        if not validation_result.is_valid:
            if validation_result.suggested_correction:
                return validation_result.suggested_correction
            return utils.get_language_message(detected_lang, "no_info")
        
        # Si le score de confiance est faible mais au-dessus du seuil, ajouter une note
        if validation_result.confidence_score < 0.8:
            answer = answer.strip()
            note = utils.get_language_message(detected_lang, "uncertainty_note")
            return f"{answer}\n\n{note}"
        
        return answer.strip()
        
    except Exception as e:
        logger.error(f"Erreur dans post_process_answer: {str(e)}")
        return utils.get_language_message(detected_lang, "fallback")

# === Dictionnaire de questions suggérées par thème ===
QUESTIONS_PAR_THEME = {
    "prêt": [
        "Quels sont les types de prêts disponibles ?",
        "Quel est le taux d'intérêt moyen pour un prêt immobilier ?",
        "Comment fonctionne un prêt à taux fixe ?"
    ],
    "épargne": [
        "Quels sont les meilleurs produits d'épargne en 2024 ?",
        "Comment fonctionne un livret A ?",
        "Quelle est la différence entre PEL et CEL ?"
    ],
    "crédit": [
        "Comment améliorer mon score de crédit ?",
        "Quelle est la différence entre un crédit renouvelable et un crédit à la consommation ?",
        "Quels documents fournir pour une demande de crédit ?"
    ],
    "assurance": [
        "Qu'est-ce qu'une assurance vie ?",
        "Comment fonctionne l'assurance emprunteur ?",
        "Quelles sont les garanties essentielles d'une assurance habitation ?"
    ],
    "leasing": [
        "Quelle est la différence entre LOA et LLD ?",
        "Quels sont les avantages fiscaux du leasing ?",
        "Comment calculer le coût total d'un contrat de leasing ?"
    ],
    "renting": [
        "Quels sont les avantages du renting automobile ?",
        "Que comprend un contrat de renting ?",
        "Comment est calculé le loyer en renting ?"
    ]
}

# === Configuration de la page Streamlit ===
st.set_page_config(
    page_title=f"TeamWill Finance Assistant v{config.VERSION}",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': f"TeamWill Finance Assistant v{config.VERSION} - Chatbot optimisé pour la finance"
    }
)

# === Styles CSS personnalisés pour optimiser l'interface ===
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .chat-message.user {
        background-color: #F0F2F6;
    }
    .chat-message.bot {
        background-color: #E3F2FD;
    }
    .chat-message .avatar {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 0.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    .chat-message .user-avatar {
        background-color: #6C63FF;
        color: white;
    }
    .chat-message .bot-avatar {
        background-color: #FF6584;
        color: white;
    }
    .chat-message .content {
        flex-grow: 1;
        overflow-x: auto;
    }
    .theme-selector {
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 1rem;
    }
    .sidebar-title {
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #9E9E9E;
    }
    .theme-pill {
        background-color: #E3F2FD;
        color: #1E88E5;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        cursor: pointer;
        transition: background-color 0.2s ease-in-out;
    }
    .theme-pill:hover, .theme-pill.active {
        background-color: #1E88E5;
        color: white;
    }
    /* Optimisations pour la performance */
    .stButton > button {
        transition: none !important;
    }
    .stMarkdown p {
        margin-bottom: 0;
    }
    div.element-container {
        margin-bottom: 0.5rem !important;
    }
    .chat-input {
        margin-top: 1rem;
    }
    /* Cache les éléments superflus */
    footer {
        display: none !important;
    }
    .viewerBadge_container__r5tak {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# === Fonction pour charger et cacher le modèle Mistral ===
@st.cache_resource(show_spinner=False, ttl=3600)
def load_llm():
    """
    Charge le modèle LLM de manière optimisée avec gestion intelligente des erreurs.
    """
    try:
        with st.spinner("Chargement du modèle..."):
            # Vérifier les ressources système
            system_resources = utils.monitor_system_resources()
            
            if 'error' in system_resources:
                st.error(f"❌ {system_resources['error']}")
                for rec in system_resources.get('recommendations', []):
                    st.warning(f"⚠️ {rec}")
                return None
                
            # Sélection intelligente du modèle
            memory_total_gb = system_resources['status']['memory_total_gb']
            memory_used_gb = system_resources['status']['memory_used_gb']
            memory_available_gb = memory_total_gb - memory_used_gb
            
            model_configs = {
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
            
            # Sélectionner le modèle approprié
            selected_config = None
            for config in ['large', 'medium', 'small']:
                if memory_available_gb >= model_configs[config]['min_memory']:
                    selected_config = model_configs[config]
                    break
            
            if not selected_config:
                st.error("❌ Mémoire insuffisante pour charger un modèle")
                st.warning("⚠️ Libérez de la mémoire et réessayez")
                return None
            
            # Vérifier l'existence du modèle
            model_path = Path(config.MODELS_DIR) / selected_config['name']
            if not model_path.exists():
                st.error(f"❌ Modèle non trouvé: {model_path}")
                
                # Chercher des alternatives
                available_models = list(Path(config.MODELS_DIR).glob("*.gguf"))
                if available_models:
                    model_path = available_models[0]
                    st.warning(f"⚠️ Utilisation du modèle alternatif: {model_path.name}")
                else:
                    st.error("❌ Aucun modèle disponible !")
                    return None
            
            # Configuration optimisée
            model_config = {
                'model_path': str(model_path),
                'n_ctx': selected_config['n_ctx'],
                'n_batch': selected_config['n_batch'],
                'n_threads': min(psutil.cpu_count() // 2, 4),
                'verbose': False
            }
            
            # Charger le modèle
            st.info(f"🔄 Chargement de {model_path.name}...")
            llm = LlamaCpp(**model_config)
            
            # Vérifier le chargement
            if llm:
                st.success("✅ Modèle chargé avec succès !")
                return llm
            else:
                st.error("❌ Échec du chargement du modèle")
                return None
                
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Erreur lors du chargement du modèle: {error_msg}")
        
        # Messages d'aide spécifiques
        if "Could not load Llama model" in error_msg:
            st.error("🔧 Solutions possibles:")
            st.markdown("""
            - Vérifiez que le fichier modèle n'est pas corrompu
            - Redémarrez l'application
            - Vérifiez l'espace disque disponible
            - Re-téléchargez le modèle
            """)
        elif "WinError 6" in error_msg:
            st.error("🔧 Erreur Windows:")
            st.markdown("""
            - Redémarrez l'application
            - Fermez les applications consommatrices de mémoire
            - Vérifiez les permissions des fichiers
            """)
        
        return None

# === Initialiser la mémoire pour les conversations ===
@st.cache_resource(show_spinner=False)
def initialize_memory():
    """Initialise et met en cache la mémoire de conversation."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5  # Limiter l'historique aux 5 derniers messages pour plus d'efficacité
    )

# === Initialisation de la chaîne QA ===
@st.cache_resource(ttl=1800)  # Recharge la chaîne toutes les 30 minutes
def get_qa_chain(_retriever, detected_lang="fr"):
    """
    Crée et configure la chaîne de question-réponse avec mise en cache temporaire.
    
    Args:
        _retriever: Le retriever à utiliser pour la recherche de documents
        detected_lang: La langue détectée
        
    Returns:
        La chaîne QA configurée ou None en cas d'erreur
    """
    if not _retriever:
        return None
        
    llm = load_llm()
    if not llm:
        return None
        
    memory = initialize_memory()
    
    # Configuration du prompt système avec la langue
    from langchain.prompts import PromptTemplate
    
    # Instructions de langue spécifiques
    language_instructions = {
        'fr': "Réponds en français uniquement:",
        'en': "Answer in English only:",
        'es': "Responde en español únicamente:",
        'de': "Antworte nur auf Deutsch:",
        'it': "Rispondi solo in italiano:",
        'ar': "أجب باللغة العربية فقط:"
    }
    
    response_instruction = language_instructions.get(detected_lang, language_instructions['fr'])
    
    system_template = f"""CONTEXTE:
{{context}}

QUESTION: {{question}}

INSTRUCTIONS: {response_instruction} Utilise uniquement les informations du contexte ci-dessus. Si l'information n'y est pas, dis "Je n'ai pas cette information". Réponds en 1-2 phrases courtes.

RÉPONSE:"""
    
    prompt = PromptTemplate(
        template=system_template,
        input_variables=["context", "question"]
    )
    
    # Créer la chaîne de conversation avec contexte
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_retriever,
        memory=memory,
        return_source_documents=True,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain

# === Fonction pour traiter les messages utilisateur ===
@utils.measure_response_time
def process_user_message(user_message, theme=None):
    """
    Traite le message de l'utilisateur de manière optimisée.
    
    Args:
        user_message: Message de l'utilisateur
        theme: Thème optionnel pour le filtrage
        
    Returns:
        tuple: (réponse, sources, métadonnées)
    """
    try:
        # Vérifier si c'est une demande de recommandation
        if any(keyword in user_message.lower() for keyword in ['recommand', 'conseil', 'suggér', 'quel financement', 'quelle solution']):
            response, next_state = handle_personalized_recommendation(user_message)
            return response, [], {'recommendation_state': next_state}
            
        # Vérifier les ressources avant de traiter
        resources = utils.monitor_system_resources()
        if resources.get('is_critical'):
            st.warning("⚠️ Ressources système critiques")
            for rec in resources.get('recommendations', []):
                st.info(f"💡 {rec}")
        
        # Prétraitement du message
        cleaned_message = utils.clean_user_input(user_message)
        if not cleaned_message:
            return "Je n'ai pas compris votre message. Pouvez-vous le reformuler ?", [], {}
            
        # Détection de la langue
        detected_lang = utils.detect_language_smart(cleaned_message)
        
        # Vérification de la longueur
        if len(cleaned_message) > config.MAX_INPUT_LENGTH:
            return utils.get_language_message(detected_lang, "too_long"), [], {}
            
        # Recherche des documents pertinents
        try:
            relevant_docs = utils.search_relevant_documents(
                cleaned_message,
                theme=theme,
                top_k=config.SEARCH_TOP_K,
                threshold=config.SIMILARITY_THRESHOLD
            )
        except Exception as search_error:
            st.error(f"❌ Erreur de recherche: {str(search_error)}")
            relevant_docs = []
        
        # Si aucun document pertinent
        if not relevant_docs:
            st.warning("⚠️ Aucune information pertinente trouvée")
            return utils.get_language_message(detected_lang, "no_info"), [], {}
        
        # Préparation du contexte
        context = utils.prepare_context(relevant_docs, max_length=config.MAX_CONTEXT_LENGTH)
        
        # Génération de la réponse
        try:
            llm = load_llm()
            if not llm:
                return utils.get_language_message(detected_lang, "error"), [], {}
                
            # Construire le prompt
            prompt = utils.build_prompt(
                question=cleaned_message,
                context=context,
                language=detected_lang
            )
            
            # Générer la réponse avec timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Génération de réponse trop longue")
            
            # Définir un timeout de 30 secondes
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                response = llm(prompt)
                signal.alarm(0)  # Désactiver le timeout
            except TimeoutError:
                st.warning("⚠️ Temps de réponse dépassé")
                return utils.get_language_message(detected_lang, "timeout"), [], {}
            
            # Post-traitement de la réponse
            cleaned_response = utils.clean_model_output(response)
            cleaned_response = post_process_answer(
                cleaned_response,
                relevant_docs,
                detected_lang,
                question=cleaned_message
            )
            
            # Vérification de la qualité
            if len(cleaned_response) < config.MIN_RESPONSE_LENGTH:
                st.warning("⚠️ Réponse trop courte")
                return utils.get_language_message(detected_lang, "fallback"), [], {}
            
            # Préparer les métadonnées
            metadata = {
                'detected_language': detected_lang,
                'processing_time': utils.get_current_processing_time(),
                'model_name': Path(llm.model_path).name,
                'theme': theme,
                'sources_count': len(relevant_docs)
            }
            
            # Logger l'interaction
            utils.log_interaction(
                question=user_message,
                answer=cleaned_response,
                sources=relevant_docs,
                metadata=metadata
            )
            
            return cleaned_response, relevant_docs, metadata
            
        except Exception as gen_error:
            st.error(f"❌ Erreur de génération: {str(gen_error)}")
            return utils.get_language_message(detected_lang, "error"), [], {}
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Erreur lors du traitement: {error_msg}")
        
        # Gestion spécifique des erreurs
        if "WinError 6" in error_msg:
            st.info("🔄 Erreur temporaire - Réessayez dans quelques secondes")
        elif "model" in error_msg.lower():
            st.error("🤖 Problème de modèle - Redémarrez l'application")
        elif "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
            st.error("💾 Mémoire insuffisante - Fermez d'autres applications")
        
        # Message d'erreur dans la langue appropriée
        try:
            detected_lang = utils.detect_language_smart(user_message)
            error_msg = utils.get_language_message(detected_lang, "error")
        except:
            error_msg = "Une erreur s'est produite. Veuillez réessayer. / An error occurred. Please try again."
            
        return error_msg, [], {}

# === Fonction pour évaluer les performances du chatbot ===
def evaluate_chatbot_performance():
    """
    Évalue les performances du chatbot avec la méthode simplifiée (rapide et fiable).
    
    Returns:
        Un dictionnaire contenant les résultats d'évaluation
    """
    
    try:
        # Utiliser l'évaluation simplifiée (la seule qui marche bien)
        import subprocess
        import sys
        
        result = subprocess.run([sys.executable, "Ragas/eval_simple_metrics.py"], 
                              capture_output=True, text=True, encoding='utf-8')
        
        # Récupérer les résultats les plus récents
        evaluation_dir = Path("evaluations")
        evaluation_files = list(evaluation_dir.glob("eval_simple_*.json"))
        
        # Vérifier si on a des résultats même en cas de warnings
        if result.returncode != 0 and not evaluation_files:
            return {"error": f"Erreur lors de l'évaluation: {result.stderr or result.stdout}"}
        
        if not evaluation_files:
            return {
                "error": "Aucun résultat d'évaluation disponible. Vérifiez que vous avez des interactions enregistrées."
            }
        
        # Charger le résultat le plus récent
        latest_evaluation = max(evaluation_files, key=os.path.getmtime)
        with open(latest_evaluation, "r", encoding="utf-8") as f:
            import json
            results = json.load(f)
        
        # Ajouter des métadonnées
        results["evaluation_method"] = "simple"
        results["evaluation_file"] = str(latest_evaluation)
        
        return results
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {
            "error": f"Une erreur s'est produite lors de l'évaluation: {str(e)}"
        }

# === Fonction pour afficher les visualisations ===
def show_visualizations():
    """Affiche les visualisations des performances du chatbot."""
    st.subheader("📊 Visualisations des performances")
    
    # Récupérer les fichiers de visualisation
    visualisation_dir = Path("visualisations")
    visualisation_files = list(visualisation_dir.glob("*.png"))
    
    if not visualisation_files:
        st.info("Aucune visualisation disponible. Exécutez une évaluation pour générer des visualisations.")
        return
    
    # Regrouper les visualisations par type
    ragas_trends = [f for f in visualisation_files if "trends" in f.name]
    ragas_global = [f for f in visualisation_files if "global" in f.name]
    ragas_distribution = [f for f in visualisation_files if "distribution" in f.name]
    
    # Afficher les visualisations
    cols = st.columns(3)
    
    with cols[0]:
        if ragas_trends:
            latest_trends = max(ragas_trends, key=os.path.getmtime)
            st.image(str(latest_trends), caption="Évolution des métriques RAGAS")
    
    with cols[1]:
        if ragas_global:
            latest_global = max(ragas_global, key=os.path.getmtime)
            st.image(str(latest_global), caption="Score RAGAS global")
    
    with cols[2]:
        if ragas_distribution:
            latest_distribution = max(ragas_distribution, key=os.path.getmtime)
            st.image(str(latest_distribution), caption="Distribution des scores RAGAS")

# === Fonction pour afficher les statistiques d'utilisation ===
def show_usage_statistics():
    """Affiche les statistiques d'utilisation du chatbot."""
    st.subheader("📈 Statistiques d'utilisation")
    
    # Analyser les logs
    stats = utils.analyze_logs()
    
    if "error" in stats:
        st.info(f"Statistiques non disponibles: {stats['error']}")
        return
    
    # Afficher les métriques clés
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Interactions totales</div>
        </div>
        """.format(stats["total_interactions"]), unsafe_allow_html=True)
    
    with cols[1]:
        avg_response_length = round(stats["avg_response_length"])
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Longueur moyenne des réponses (mots)</div>
        </div>
        """.format(avg_response_length), unsafe_allow_html=True)
    
    with cols[2]:
        num_sources = len(stats["sources_frequency"])
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Sources utilisées</div>
        </div>
        """.format(num_sources), unsafe_allow_html=True)
    
    with cols[3]:
        num_themes = len(stats["questions_by_theme"])
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Thèmes abordés</div>
        </div>
        """.format(num_themes), unsafe_allow_html=True)
    
    # Afficher les graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution par thème")
        theme_data = stats["questions_by_theme"]
        if theme_data:
            themes = list(theme_data.keys())
            counts = list(theme_data.values())
            fig = px.bar(
                x=themes, 
                y=counts, 
                labels={"x": "Thème", "y": "Nombre de questions"},
                color=counts,
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pas assez de données pour afficher la distribution par thème.")
    
    with col2:
        st.subheader("Interactions par heure")
        hours_data = stats["interactions_by_hour"]
        hours = list(map(int, hours_data.keys()))
        interactions = list(hours_data.values())
        df = pd.DataFrame({"Heure": hours, "Interactions": interactions})
        df = df.sort_values("Heure")
        fig = px.line(
            df, 
            x="Heure", 
            y="Interactions",
            markers=True,
            line_shape="spline"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sources les plus utilisées
    st.subheader("Sources les plus utilisées")
    sources_data = stats["sources_frequency"]
    if sources_data:
        sources = list(sources_data.keys())
        counts = list(sources_data.values())
        
        # Créer un DataFrame pour le tri
        df = pd.DataFrame({"Source": sources, "Fréquence": counts})
        df = df.sort_values("Fréquence", ascending=False).head(10)  # Top 10
        
        fig = px.bar(
            df,
            x="Fréquence", 
            y="Source",
            orientation="h",
            color="Fréquence",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas assez de données pour afficher les sources les plus utilisées.")

# === Interface pour ajouter de nouveaux documents ===
def document_management():
    """Interface pour gérer les documents dans la base de connaissances."""
    st.title("📚 Gestion des documents")
    
    # Liste des documents actuels
    st.subheader("Documents actuels")
    docs_dir = Path(config.DOCUMENTS_DIR)
    docs = list(docs_dir.glob("*.*"))
    
    if not docs:
        st.info("Aucun document dans la base de connaissances.")
    else:
        # Créer un DataFrame avec les documents et leurs informations
        docs_info = []
        for doc in docs:
            size_kb = round(doc.stat().st_size / 1024, 1)
            last_modified = dt.fromtimestamp(doc.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            docs_info.append({
                "Nom": doc.name,
                "Taille (KB)": size_kb,
                "Dernière modification": last_modified,
                "Chemin": str(doc)
            })
        
        docs_df = pd.DataFrame(docs_info)
        st.dataframe(docs_df)
    
    # Interface pour ajouter un nouveau document
    st.subheader("Ajouter un nouveau document")
    
    uploaded_file = st.file_uploader("Choisir un fichier", type=["pdf", "txt", "docx"])
    
    if uploaded_file:
        # Sauvegarder le fichier temporairement
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        st.info(f"Fichier téléchargé: {uploaded_file.name}")
        
        # Bouton pour ajouter à la base de connaissances
        if st.button("Ajouter à la base de connaissances"):
            with st.spinner("Traitement du document..."):
                # Copier le fichier dans le répertoire documents
                target_path = docs_dir / uploaded_file.name
                import shutil
                shutil.copy(tmp_path, target_path)
                
                # Ajouter à la base vectorielle
                result = utils.add_document(target_path)
                st.success(result)
                
                # Supprimer le fichier temporaire
                os.unlink(tmp_path)
                st.rerun()  # Rafraîchir l'interface

# === Interface d'évaluation ===
def evaluation_interface():
    """Interface pour évaluer les performances du chatbot avec l'évaluateur simplifié."""
    st.title("🎯 Évaluation RAG")
    st.markdown("**Évaluation professionnelle avec métriques simplifiées**")
    
    # Import du système d'évaluation simplifié
    try:
        from Ragas.eval_rag import SimpleRAGEvaluator
    except ImportError:
        st.error("❌ Module d'évaluation non trouvé. Vérifiez l'installation.")
        return
    
    # Initialiser l'état de session pour les résultats
    if "simple_rag_results" not in st.session_state:
        st.session_state.simple_rag_results = None
    
    # Information sur l'évaluation
    st.info("""
    🔬 **Évaluation RAG Simplifiée**
    
    Métriques équivalentes à RAGAS mais calculées de manière algorithmique :
    - **🎯 Faithfulness** : Fidélité aux sources (pas d'hallucinations)
    - **📝 Answer Relevancy** : Pertinence par rapport à la question
    - **🎯 Context Precision** : Qualité du contexte récupéré
    - **🔍 Context Recall** : Couverture complète du contexte pertinent
    
    ✅ **Avantages :** Rapide, fiable, sans dépendances externes complexes
    """)
    
    # Configuration
    max_samples = st.number_input(
        "📊 Nombre max d'interactions à analyser",
        min_value=5,
        max_value=500,
        value=100,
        help="Nombre d'interactions à analyser pour l'évaluation"
    )
    
    # Bouton d'évaluation
    if st.button("🚀 Lancer l'évaluation RAG", type="primary", use_container_width=True):
        
        evaluator = SimpleRAGEvaluator()
        
        # Récupération des logs
        log_file = evaluator.get_latest_log_file()
        if not log_file:
            st.error("❌ Aucun fichier de log trouvé. Utilisez d'abord le chatbot pour avoir des interactions à analyser.")
            st.stop()
        
        st.info(f"📂 Analyse du fichier: {log_file.name}")
        
        # Préparation du dataset
        samples = evaluator.prepare_dataset(log_file, max_samples)
        if not samples:
            st.stop()
        
        # Évaluation
        results = evaluator.run_evaluation(samples)
        if not results:
            st.stop()
        
        # Affichage et sauvegarde
        evaluator.display_results(results)
        evaluator.save_results(results, len(samples))
        
        # Sauvegarder dans session state
        st.session_state.simple_rag_results = {
            "results": results,
            "dataset_size": len(samples),
            "timestamp": dt.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        st.balloons()
    
    # Afficher les derniers résultats s'ils existent
    if st.session_state.simple_rag_results:
        st.markdown("---")
        st.subheader("📊 Derniers résultats")
        
        rag_data = st.session_state.simple_rag_results
        st.info(f"📊 {rag_data['dataset_size']} interactions analysées | 🕐 {rag_data['timestamp']}")
        
        # Bouton pour effacer
        if st.button("🗑️ Effacer les résultats", type="secondary"):
            st.session_state.simple_rag_results = None
            st.rerun()
    
    # Historique des évaluations
    st.markdown("---")
    st.subheader("📋 Historique des évaluations")
    
    evaluation_dir = Path("evaluations")
    if evaluation_dir.exists():
        # Lister les évaluations simplifiées
        simple_files = list(evaluation_dir.glob("simple_eval_*.json"))
        simple_files.sort(key=os.path.getmtime, reverse=True)
        
        if simple_files:
            for eval_file in simple_files[:5]:  # 5 plus récentes
                try:
                    with open(eval_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    timestamp = data.get("timestamp", "").replace("_", " ").replace("-", "/")
                    size = data.get("dataset_size", 0)
                    
                    # Calculer score moyen
                    metrics = data.get("metrics", {})
                    if isinstance(metrics, dict) and metrics:
                        avg_score = sum(metrics.values()) / len(metrics)
                        score_text = f"Score: {avg_score:.3f}"
                        
                        if avg_score >= 0.8:
                            icon = "🟢"
                        elif avg_score >= 0.6:
                            icon = "🟡"
                        elif avg_score >= 0.4:
                            icon = "🟠"
                        else:
                            icon = "🔴"
                    else:
                        icon = "⚪"
                        score_text = "Score: N/A"
                    
                    st.markdown(f"{icon} **{timestamp}** - {size} interactions - {score_text}")
                    
                except Exception:
                    continue
        else:
            st.write("Aucune évaluation trouvée. Lancez votre première évaluation !")
    else:
        st.write("Aucune évaluation trouvée. Lancez votre première évaluation !")

# === Interface principale du chatbot ===
def chatbot_interface():
    """Interface principale pour interagir avec le chatbot."""
    # En-tête
    st.markdown("<h1 class='main-header'>💼 Assistant TeamWill Finance</h1>", unsafe_allow_html=True)
    st.markdown("Posez vos questions sur les produits et services financiers.")
    
    # Initialiser la session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "theme" not in st.session_state:
        st.session_state.theme = "Tous"
    
    # Sélecteur de thème
    st.markdown("<div class='theme-selector'>", unsafe_allow_html=True)
    cols = st.columns([1, 3])
    with cols[0]:
        st.write("Filtrer par thème:")
    with cols[1]:
        theme_html = ""
        for theme in config.THEMES:
            active_class = "active" if theme == st.session_state.theme else ""
            description = config.THEME_DESCRIPTIONS.get(theme, theme)
            onclick = f"Streamlit.setComponentValue('{theme}')"
            theme_html += f'<span class="theme-pill {active_class}" title="{description}" onclick="{onclick}">{theme}</span> '
        
        selected_theme = st.selectbox(
            "Thème:",
            options=config.THEMES,
            index=config.THEMES.index(st.session_state.theme) if st.session_state.theme in config.THEMES else 0,
            label_visibility="collapsed"
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
    
    # Afficher les questions suggérées dans la sidebar si un thème est sélectionné
    if selected_theme != "Tous" and selected_theme in QUESTIONS_PAR_THEME:
        st.sidebar.markdown(f"### Questions suggérées - {selected_theme}")
        
        for question in QUESTIONS_PAR_THEME[selected_theme]:
            if st.sidebar.button(question, key=f"btn_{question}", use_container_width=True):
                # Ajouter la question à l'historique et la traiter
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Obtenir la réponse du chatbot
                with st.spinner("Réflexion en cours..."):
                    response, sources, metadata = process_user_message(question, theme=st.session_state.theme)
                
                # Ajouter la réponse à l'historique
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response,
                    "sources": sources,
                    "metadata": metadata
                })
                
                # Forcer un rechargement de la page
                st.rerun()
    
    # Container pour l'historique des messages avec hauteur limitée
    chat_container = st.container()
    
    # Container pour la saisie utilisateur fixé en bas
    input_container = st.container()
    
    # Afficher les messages précédents
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="avatar user-avatar">👤</div>
                    <div class="content">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Afficher uniquement le contenu de la réponse, sans sources ni temps de réponse
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="content">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Zone de saisie utilisateur
    with input_container:
        if prompt := st.chat_input("Posez votre question ici...", key="chat_input"):
            # Ajouter le message utilisateur à l'historique
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Obtenir la réponse du chatbot
            with st.spinner("Réflexion en cours..."):
                response, sources, metadata = process_user_message(prompt, theme=st.session_state.theme)
            
            # Ajouter la réponse à l'historique
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources,
                "metadata": metadata
            })
            
            # Forcer un rechargement de la page
            st.rerun()
    
    # Bouton pour effacer l'historique
    if st.session_state.messages and st.sidebar.button("Effacer la conversation", type="secondary", use_container_width=True):
        st.session_state.messages = []
        # Réinitialiser la mémoire de conversation
        initialize_memory()
        st.rerun()

# === Navigation principale ===
def main():
    """Interface principale de l'application."""
    # Vérification des ressources système et auto-configuration
    if "optimal_config" not in st.session_state:
        optimal_config = utils.get_optimal_config()
        st.session_state.optimal_config = optimal_config
        
        # Mettre à jour la configuration avec les valeurs optimales
        if optimal_config:
            # Mettre à jour la configuration du modèle si nécessaire
            if 'model_size' in optimal_config:
                model_path = config.MODELS_DIR / optimal_config['model_size']
                if model_path.exists():
                    config.MODEL_CONFIG['model_path'] = str(model_path)
            
            # Mettre à jour les autres paramètres
            for param in ['n_threads', 'n_ctx', 'n_batch']:
                if param in optimal_config:
                    config.MODEL_CONFIG[param] = optimal_config[param]
            
            # Mettre à jour la configuration du text splitter
            if 'chunk_size' in optimal_config:
                config.TEXT_SPLITTER_CONFIG['chunk_size'] = optimal_config['chunk_size']
            if 'chunk_overlap' in optimal_config:
                config.TEXT_SPLITTER_CONFIG['chunk_overlap'] = optimal_config['chunk_overlap']
    
    # Barre latérale
    st.sidebar.image("img/teamwill.png", width=250)
    st.sidebar.title("Teamwill Finance Assistant")
    st.sidebar.markdown(f"Version {config.VERSION}")
    
    # Afficher les informations système dans la sidebar (caché par défaut)
    system_info = utils.monitor_system_resources()
    with st.sidebar.expander("Informations système", expanded=False):
        st.write("CPU:", f"{system_info.get('cpu_percent', 'N/A')}% - {system_info.get('cpu_freq', 'N/A')}")
        st.write("Mémoire:", f"{system_info.get('memory_percent', 'N/A')}% ({system_info.get('memory_used_gb', 0):.1f}/{system_info.get('memory_total_gb', 0):.1f} GB)")
        st.write("Disque:", f"{system_info.get('disk_percent', 'N/A')}% ({system_info.get('disk_used_gb', 0):.1f}/{system_info.get('disk_total_gb', 0):.1f} GB)")
        st.write("Processus Python:", f"{system_info.get('process_memory_gb', 0):.2f} GB")
        
        # Afficher la configuration utilisée
        st.write("**Configuration:**")
        st.write(f"Modèle: {Path(config.MODEL_CONFIG['model_path']).name}")
        st.write(f"Threads: {config.MODEL_CONFIG['n_threads']}")
        st.write(f"Contexte: {config.MODEL_CONFIG['n_ctx']}")
        st.write(f"Batch: {config.MODEL_CONFIG['n_batch']}")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        options=["Chatbot", "Statistiques", "Gestion des documents", "Évaluation"]
    )
    
    # Afficher la page sélectionnée
    if page == "Chatbot":
        chatbot_interface()
    elif page == "Statistiques":
        show_usage_statistics()
    elif page == "Gestion des documents":
        document_management()
    elif page == "Évaluation":
        evaluation_interface()

def handle_personalized_recommendation(message):
    """
    Gère le processus de recommandation personnalisée en collectant les informations nécessaires.
    
    Args:
        message: Message initial de l'utilisateur
    
    Returns:
        tuple: (réponse, étape suivante)
    """
    # États possibles du processus de recommandation
    RECOMMENDATION_STATES = {
        'initial': "Pour vous faire une recommandation personnalisée, j'ai besoin de quelques informations. Quel est votre budget mensuel maximum pour le financement ?",
        'budget': "Pour quelle utilisation souhaitez-vous le véhicule ? (personnel, professionnel, mixte)",
        'usage': "Sur quelle durée souhaitez-vous financer le véhicule ? (en années)",
        'duree': "Quelle est votre situation professionnelle ? (salarié en CDI, indépendant, entreprise)",
        'situation': None  # État final
    }
    
    # Vérifier si un processus de recommandation est en cours
    if 'recommendation_state' not in st.session_state:
        st.session_state.recommendation_state = 'initial'
        st.session_state.client_info = {}
        return RECOMMENDATION_STATES['initial'], 'budget'
    
    current_state = st.session_state.recommendation_state
    
    # Traiter la réponse selon l'état actuel
    if current_state == 'budget':
        try:
            budget = float(''.join(filter(str.isdigit, message)))
            st.session_state.client_info['budget_mensuel'] = budget
            st.session_state.recommendation_state = 'usage'
            return RECOMMENDATION_STATES['usage'], 'usage'
        except:
            return "Je n'ai pas compris le montant. Pouvez-vous indiquer votre budget mensuel en chiffres ?", 'budget'
    
    elif current_state == 'usage':
        usage = message.lower()
        if any(u in usage for u in ['pro', 'travail', 'entreprise']):
            usage = 'professionnel'
        elif any(u in usage for u in ['perso', 'personnel', 'famille']):
            usage = 'personnel'
        else:
            usage = 'mixte'
        st.session_state.client_info['usage'] = usage
        st.session_state.recommendation_state = 'duree'
        return RECOMMENDATION_STATES['duree'], 'duree'
    
    elif current_state == 'duree':
        try:
            duree = int(''.join(filter(str.isdigit, message)))
            st.session_state.client_info['duree_souhaitee'] = duree
            st.session_state.recommendation_state = 'situation'
            return RECOMMENDATION_STATES['situation'], 'situation'
        except:
            return "Je n'ai pas compris la durée. Pouvez-vous indiquer le nombre d'années en chiffres ?", 'duree'
    
    elif current_state == 'situation':
        situation = message.lower()
        if 'cdi' in situation or 'salari' in situation:
            situation = 'salarie'
        elif 'indep' in situation:
            situation = 'independant'
        elif 'entreprise' in situation:
            situation = 'entreprise'
        else:
            situation = 'autre'
        st.session_state.client_info['situation'] = situation
        
        # Générer la recommandation
        recommendations = utils.analyze_client_profile(st.session_state.client_info)
        
        # Réinitialiser l'état
        st.session_state.recommendation_state = 'initial'
        st.session_state.client_info = {}
        
        return utils.format_recommendation_response(recommendations), None
    
    return "Je n'ai pas compris votre réponse. Pouvez-vous reformuler ?", current_state

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Une erreur s'est produite: {str(e)}")
        
        # Messages d'aide spécifiques
        error_msg = str(e)
        if "WinError 6" in error_msg:
            st.info("🔄 **Solution:** Redémarrez Streamlit et réessayez")
        elif "model" in error_msg.lower():
            st.info("🤖 **Solution:** Vérifiez la configuration du modèle dans config.py")
        
        # Bouton pour redémarrer
        if st.button("🔄 Redémarrer l'application"):
            st.rerun() 