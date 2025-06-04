"""
Fonctions utilitaires pour le chatbot TeamWill.
"""

import json
import time
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

import config
import requests
from requests.exceptions import RequestException
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import streamlit as st
import signal
from functools import wraps

# Charger les variables d'environnement
load_dotenv()

# === Configuration CARAPI ===
CARAPI_CONFIG = {
    "api_token": os.getenv("CARAPI_TOKEN", "bd7adcd2-beb0-483b-9bb8-a72cbb27de52"),
    "api_secret": os.getenv("CARAPI_SECRET", "1b1d5c0761e33562b8680f44cf001cd3"),
    "base_url": "https://carapi.app/api",  # URL de base corrigée selon la doc
    "jwt_cache_file": ".cache/.carapi_jwt"  # Déplacé vers un dossier cache dédié
}

# === Journalisation des interactions ===
def setup_logging():
    """Configure le système de logging avec rotation des fichiers."""
    import logging
    from logging.handlers import RotatingFileHandler
    import time
    
    # Créer le répertoire des logs si nécessaire
    log_dir = Path(config.LOGS_DIR)
    log_dir.mkdir(exist_ok=True)
    
    # Configuration du logger principal
    logger = logging.getLogger('teamwill_chatbot')
    logger.setLevel(logging.INFO)
    
    # Formatter pour les logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler pour les logs d'erreur
    error_handler = RotatingFileHandler(
        log_dir / 'error.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    # Handler pour les logs d'information
    info_handler = RotatingFileHandler(
        log_dir / 'info.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    
    # Handler pour les logs de performance
    perf_handler = RotatingFileHandler(
        log_dir / 'performance.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(formatter)
    
    # Ajouter les handlers au logger
    logger.addHandler(error_handler)
    logger.addHandler(info_handler)
    logger.addHandler(perf_handler)
    
    return logger

# Logger global
logger = setup_logging()

def log_interaction(question, answer, sources, metadata=None):
    """
    Enregistre une interaction avec le chatbot de manière structurée.
    
    Args:
        question: Question de l'utilisateur
        answer: Réponse du chatbot
        sources: Sources utilisées
        metadata: Métadonnées additionnelles
    """
    try:
        # Créer le répertoire des logs si nécessaire
        log_dir = Path(config.LOGS_DIR)
        log_dir.mkdir(exist_ok=True)
        
        # Nom du fichier basé sur la date
        timestamp = time.strftime("%Y%m%d")
        log_file = log_dir / f"interactions_{timestamp}.jsonl"
        
        # Préparer les données
        metadata = metadata or {}
        metadata.update({
            'timestamp': time.time(),
            'datetime': time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        # Ajouter les métriques de performance
        if hasattr(st.session_state, 'current_processing_time'):
            metadata['processing_time'] = st.session_state.current_processing_time
        
        # Ajouter les ressources système
        try:
            resources = monitor_system_resources()
            if isinstance(resources, dict) and 'error' not in resources:
                metadata['system_resources'] = resources['status']
        except:
            pass
        
        # Créer l'entrée de log
        log_entry = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'metadata': metadata
        }
        
        # Écrire dans le fichier JSONL
        with open(log_file, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')
            
        # Logger les métriques de performance
        if 'processing_time' in metadata:
            logger.info(f"Performance - Temps de traitement: {metadata['processing_time']:.2f}s")
            
        # Logger les alertes système
        if 'system_resources' in metadata:
            resources = metadata['system_resources']
            if resources['memory_percent'] > config.MONITORING_CONFIG['memory_warning']:
                logger.warning(f"Alerte mémoire - Utilisation: {resources['memory_percent']}%")
            if resources['cpu_percent'] > config.MONITORING_CONFIG['cpu_warning']:
                logger.warning(f"Alerte CPU - Utilisation: {resources['cpu_percent']}%")
                
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement de l'interaction: {str(e)}")
        
def clean_old_logs():
    """Nettoie les anciens fichiers de log selon la politique de rétention."""
    try:
        log_dir = Path(config.LOGS_DIR)
        retention_days = 7  # Garder 7 jours de logs
        
        current_time = time.time()
        for log_file in log_dir.glob("*.log*"):
            file_time = log_file.stat().st_mtime
            if current_time - file_time > retention_days * 24 * 3600:
                log_file.unlink()
                logger.info(f"Suppression du fichier de log ancien: {log_file.name}")
                
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des logs: {str(e)}")

# === Analyse des logs ===
def analyze_logs(log_file=None):
    """
    Analyse les logs d'interactions et retourne des statistiques de base.
    
    Args:
        log_file: Chemin vers le fichier de log à analyser (si None, utilise le plus récent)
        
    Returns:
        Un dictionnaire contenant les statistiques d'utilisation
    """
    # Si aucun fichier n'est spécifié, utiliser le plus récent
    if log_file is None:
        log_files = list(config.LOGS_DIR.glob("interactions_*.jsonl"))
        if not log_files:
            return {"error": "Aucun fichier de log trouvé"}
        log_file = max(log_files, key=lambda f: f.stat().st_mtime)
    
    # Statistiques à collecter
    stats = {
        "total_interactions": 0,
        "questions_by_theme": {},
        "sources_frequency": {},
        "avg_response_length": 0,
        "interactions_by_hour": {str(h): 0 for h in range(24)}
    }
    
    # Analyse du fichier de log
    total_response_length = 0
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                interaction = json.loads(line.strip())
                stats["total_interactions"] += 1
                
                # Longueur moyenne des réponses
                response_length = len(interaction["answer"].split())
                total_response_length += response_length
                
                # Distribution des sources
                for source in interaction.get("sources", []):
                    stats["sources_frequency"][source] = stats["sources_frequency"].get(source, 0) + 1
                
                # Interactions par heure
                timestamp = datetime.datetime.fromisoformat(interaction["timestamp"])
                hour = timestamp.hour
                stats["interactions_by_hour"][str(hour)] = stats["interactions_by_hour"].get(str(hour), 0) + 1
                
                # Analyse thématique (simple et basique)
                question = interaction["question"].lower()
                themes = {
                    "prêt": "pret" in question or "emprunt" in question,
                    "épargne": "epargne" in question or "placement" in question,
                    "crédit": "credit" in question,
                    "assurance": "assurance" in question,
                    "leasing": "leasing" in question or "loa" in question,
                    "renting": "renting" in question or "location" in question
                }
                
                for theme, is_present in themes.items():
                    if is_present:
                        stats["questions_by_theme"][theme] = stats["questions_by_theme"].get(theme, 0) + 1
                
            except json.JSONDecodeError:
                continue
    
    # Calculer la longueur moyenne des réponses
    if stats["total_interactions"] > 0:
        stats["avg_response_length"] = total_response_length / stats["total_interactions"]
    
    return stats

# === Création et gestion de la base vectorielle ===
def get_embeddings():
    """
    Renvoie une instance de l'embedding configuré.
    
    Returns:
        HuggingFaceEmbeddings: L'instance d'embedding configurée
    """
    return HuggingFaceEmbeddings(**config.EMBEDDING_CONFIG)

def get_text_splitter():
    """
    Renvoie une instance du text_splitter configuré.
    
    Returns:
        RecursiveCharacterTextSplitter: L'instance du text_splitter configurée
    """
    return RecursiveCharacterTextSplitter(**config.TEXT_SPLITTER_CONFIG)

def get_vectordb():
    """
    Charge la base vectorielle existante ou en crée une nouvelle si elle n'existe pas.
    
    Returns:
        Chroma: L'instance de la base vectorielle
    """
    try:
        embedding = get_embeddings()
        
        # Vérifier si la base vectorielle est initialisée
        vectordb_dir = Path(config.VECTORDB_DIR)
        if not vectordb_dir.exists() or not any(vectordb_dir.iterdir()):
            print("ATTENTION: La base vectorielle n'est pas initialisée. Utilisation d'une base vide.")
            return Chroma(persist_directory=str(vectordb_dir), embedding_function=embedding)
            
        db = Chroma(persist_directory=str(vectordb_dir), embedding_function=embedding)
        
        # Vérifier si la base contient des documents
        if db._collection.count() == 0:
            print("ATTENTION: La base vectorielle est vide.")
            
        return db
    except Exception as e:
        print(f"Erreur lors du chargement de la base vectorielle: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def add_document(file_path):
    """
    Ajoute un document à la base vectorielle existante.
    
    Args:
        file_path: Chemin vers le document à ajouter
        
    Returns:
        str: Message de confirmation ou d'erreur
    """
    try:
        # Sélectionner le bon chargeur en fonction de l'extension du fichier
        file_path = Path(file_path)
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix.lower() in [".docx", ".doc"]:
            loader = UnstructuredFileLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path))
        
        # Charger le document
        docs = loader.load()
        
        # Découper le document en chunks
        text_splitter = get_text_splitter()
        documents = text_splitter.split_documents(docs)
        
        # Ajouter des métadonnées sur la source
        for doc in documents:
            doc.metadata["source"] = file_path.name
        
        # Charger la base vectorielle
        vectordb = get_vectordb()
        
        # Ajouter les chunks à la base vectorielle
        vectordb.add_documents(documents)
        
        # Persister les modifications
        vectordb.persist()
        
        return f"Document '{file_path.name}' ajouté avec succès à la base vectorielle. {len(documents)} fragments créés."
    except Exception as e:
        return f"Erreur lors de l'ajout du document: {str(e)}"

# === Retriever thématique amélioré ===
class ThematicMMRRetriever(BaseRetriever):
    """
    Retriever optimisé avec MMR et filtrage thématique avancé.
    """
    vectorstore: Chroma
    k: int = config.RETRIEVER_CONFIG["k"]
    fetch_k: int = config.RETRIEVER_CONFIG["fetch_k"]
    lambda_mult: float = config.RETRIEVER_CONFIG["lambda_mult"]
    filter_theme: str = None
    use_compression: bool = config.RETRIEVER_CONFIG["use_compression"]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Récupère les documents pertinents avec optimisations.
        
        Args:
            query: La requête utilisateur
            
        Returns:
            Liste des documents pertinents
        """
        # 1. Enrichissement intelligent de la requête
        enhanced_query = self._enhance_query(query)
        
        # 2. Récupération des documents avec MMR
        docs = self.vectorstore.max_marginal_relevance_search(
            enhanced_query,
            k=self.k,
            fetch_k=self.fetch_k,
            lambda_mult=self.lambda_mult
        )
        
        # 3. Filtrage thématique si nécessaire
        if self.filter_theme and self.filter_theme != "Tous":
            docs = self._apply_theme_filter(docs)
        
        # 4. Compression contextuelle si activée
        if self.use_compression and len(docs) > 1:
            docs = self._compress_context(docs)
        
        return docs
    
    def _enhance_query(self, query: str) -> str:
        """
        Enrichit la requête avec des termes pertinents.
        """
        # Termes financiers par thème
        theme_terms = {
            "financement_auto": ["financement", "véhicule", "auto", "voiture"],
            "credit_auto": ["crédit", "prêt", "taux", "mensualité"],
            "leasing": ["leasing", "location", "LLD", "LOA"],
            "assurance": ["assurance", "garantie", "couverture"]
        }
        
        # Enrichir avec les termes du thème si défini
        if self.filter_theme and self.filter_theme in theme_terms:
            relevant_terms = theme_terms[self.filter_theme]
            # Ajouter uniquement les termes non présents dans la requête
            new_terms = [term for term in relevant_terms 
                        if term not in query.lower() 
                        and not any(t in query.lower() for t in term.split())]
            if new_terms:
                return f"{query} {' '.join(new_terms[:2])}"
        
        return query
    
    def _apply_theme_filter(self, docs: List[Document]) -> List[Document]:
        """
        Filtre les documents selon le thème avec scoring.
        """
        if not docs:
            return []
            
        # Mots-clés par thème
        theme_keywords = config.THEME_KEYWORDS.get(self.filter_theme, [])
        if not theme_keywords:
            return docs
            
        # Score et filtre les documents
        scored_docs = []
        for doc in docs:
            score = sum(1 for keyword in theme_keywords 
                       if keyword in doc.page_content.lower())
            if score > 0:  # Document pertinent pour le thème
                scored_docs.append((doc, score))
        
        # Trier par score et retourner les documents
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:self.k]]
    
    def _compress_context(self, docs: List[Document]) -> List[Document]:
        """
        Compresse le contexte pour plus de pertinence.
        """
        if not docs:
            return []
            
        # Fusionner tous les contenus
        all_content = " ".join(doc.page_content for doc in docs)
        
        # Extraire les phrases les plus importantes
        sentences = all_content.split('.')
        scored_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            # Score basé sur les mots-clés financiers et la longueur
            score = sum(1 for word in sentence.lower().split() 
                       if word in config.THEME_KEYWORDS.get(self.filter_theme, []))
            score -= len(sentence.split()) * 0.01  # Pénalité pour longueur
            scored_sentences.append((sentence, score))
        
        # Sélectionner les meilleures phrases
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        best_sentences = [s[0] for s in scored_sentences[:5]]  # Top 5 phrases
        
        # Créer un nouveau document compressé
        compressed_content = ". ".join(best_sentences)
        return [Document(page_content=compressed_content, metadata=docs[0].metadata)]

def get_retriever(llm=None, theme=None, use_compression=False):
    """
    Crée et configure un retriever avancé avec compression contextuelle optionnelle.
    
    Args:
        llm: Le modèle de langage à utiliser pour la compression (optionnel)
        theme: Le thème à utiliser pour le filtrage (optionnel)
        use_compression: Indique si la compression contextuelle doit être utilisée
        
    Returns:
        BaseRetriever: Le retriever configuré
    """
    # Charger la base vectorielle
    vectordb = get_vectordb()
    if not vectordb:
        return None
    
    # Créer le retriever thématique MMR
    thematic_mmr_retriever = ThematicMMRRetriever(
        vectorstore=vectordb,
        k=config.RETRIEVER_CONFIG["k"],
        fetch_k=config.RETRIEVER_CONFIG["fetch_k"],
        lambda_mult=config.RETRIEVER_CONFIG["lambda_mult"],
        filter_theme=theme
    )
    
    # Si un LLM est fourni et que la compression est activée, ajouter un compresseur de contexte
    if llm and use_compression:
        compressor = LLMChainExtractor.from_llm(llm)
        return ContextualCompressionRetriever(
            base_retriever=thematic_mmr_retriever,
            base_compressor=compressor,
            return_source_documents=True
        )
    
    return thematic_mmr_retriever

# === Mesure des performances ===
def measure_response_time(func):
    """
    Décorateur pour mesurer le temps de réponse d'une fonction.
    
    Args:
        func: La fonction à mesurer
        
    Returns:
        Une fonction encapsulée qui mesure le temps d'exécution
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Ajouter le temps d'exécution aux métadonnées si le résultat est un tuple
        if isinstance(result, tuple) and len(result) >= 2:
            response, metadata = result if len(result) == 2 else (result[0], result[-1])
            if isinstance(metadata, dict):
                metadata["response_time"] = execution_time
            return result
        
        return result, {"response_time": execution_time}
    
    return wrapper

# === Surveillance des ressources système ===
def monitor_system_resources():
    """
    Surveille et gère les ressources système de manière proactive.
    Inclut des actions automatiques pour optimiser les performances.
    
    Returns:
        Dict: État des ressources et actions prises
    """
    try:
        import psutil
        import gc
        
        # Seuils critiques
        MEMORY_CRITICAL = 90  # Pourcentage
        CPU_CRITICAL = 95     # Pourcentage
        DISK_CRITICAL = 95    # Pourcentage
        
        # Collecte des métriques
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        process = psutil.Process()
        
        # Conversion en GB pour plus de clarté
        memory_used_gb = memory.used / (1024 ** 3)
        memory_total_gb = memory.total / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)
        disk_used_gb = disk.used / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)
        process_memory_gb = process.memory_info().rss / (1024 ** 3)
        
        # Actions prises
        actions_taken = []
        
        # 1. Gestion de la mémoire critique
        if memory.percent >= MEMORY_CRITICAL:
            # Forcer le garbage collector
            gc.collect()
            actions_taken.append("Garbage collection forcée")
            
            # Réduire la taille du cache si possible
            if hasattr(st, 'cache_clear'):
                st.cache_clear()
                actions_taken.append("Cache Streamlit nettoyé")
                
            # Réduire les paramètres du modèle
            config.MODEL_CONFIG.update({
                'n_ctx': 1024,
                'n_batch': 256,
                'max_tokens': 100
            })
            actions_taken.append("Paramètres du modèle réduits")
        
        # 2. Gestion du CPU critique
        if cpu_percent >= CPU_CRITICAL:
            # Réduire le nombre de threads
            config.MODEL_CONFIG['n_threads'] = max(1, config.MODEL_CONFIG['n_threads'] - 1)
            actions_taken.append("Nombre de threads réduit")
            
            # Désactiver la compression si active
            if config.RETRIEVER_CONFIG['use_compression']:
                config.RETRIEVER_CONFIG['use_compression'] = False
                actions_taken.append("Compression contextuelle désactivée")
        
        # 3. Gestion du disque critique
        if disk.percent >= DISK_CRITICAL:
            # Nettoyer les fichiers temporaires
            clean_temp_files()
            actions_taken.append("Fichiers temporaires nettoyés")
            
            # Nettoyer les vieux logs
            clean_old_logs()
            actions_taken.append("Anciens logs nettoyés")
        
        # 4. Optimisations proactives
        if memory.percent > 70 or cpu_percent > 70:
            # Réduire la taille des chunks et le nombre de documents
            config.VECTORDB_CONFIG['chunk_size'] = 250
            config.RETRIEVER_CONFIG['k'] = 2
            config.RETRIEVER_CONFIG['fetch_k'] = 4
            actions_taken.append("Paramètres de recherche optimisés")
        
        # Construire le rapport
        status = {
            'cpu': {
                'percent': cpu_percent,
                'critical': cpu_percent >= CPU_CRITICAL,
                'warning': cpu_percent >= 70
            },
            'memory': {
                'used_gb': round(memory_used_gb, 2),
                'total_gb': round(memory_total_gb, 2),
                'available_gb': round(memory_available_gb, 2),
                'percent': memory.percent,
                'critical': memory.percent >= MEMORY_CRITICAL,
                'warning': memory.percent >= 70
            },
            'disk': {
                'used_gb': round(disk_used_gb, 2),
                'total_gb': round(disk_total_gb, 2),
                'percent': disk.percent,
                'critical': disk.percent >= DISK_CRITICAL,
                'warning': disk.percent >= 70
            },
            'process': {
                'memory_gb': round(process_memory_gb, 2),
                'cpu_percent': process.cpu_percent()
            }
        }
        
        # Générer les recommandations
        recommendations = []
        if status['memory']['warning']:
            recommendations.append("Considérer le nettoyage de la mémoire ou le redémarrage de l'application")
        if status['cpu']['warning']:
            recommendations.append("Réduire la charge de travail ou augmenter les ressources CPU")
        if status['disk']['warning']:
            recommendations.append("Libérer de l'espace disque")
        
        return {
            'status': status,
            'actions_taken': actions_taken,
            'recommendations': recommendations,
            'is_critical': any(s.get('critical', False) for s in status.values())
        }
        
    except Exception as e:
        logger.error(f"Erreur de surveillance système: {str(e)}")
        return {
            'error': str(e),
            'recommendations': ['Vérifier les permissions système', 'Installer psutil']
        }

def clean_temp_files():
    """
    Nettoie les fichiers temporaires du projet.
    """
    try:
        # Nettoyer le cache
        cache_dir = config.CACHE_DIR
        for file in cache_dir.glob("*"):
            if file.is_file():
                file.unlink()
            
        # Nettoyer les fichiers temporaires
        temp_patterns = ["*.tmp", "*.temp", "*.log.*", "*.bak"]
        for pattern in temp_patterns:
            for file in config.BASE_DIR.glob(f"**/{pattern}"):
                if file.is_file():
                    file.unlink()
                    
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des fichiers temporaires: {str(e)}")

# === Optimisation automatique ===
def get_optimal_config():
    """
    Détecte les ressources système et renvoie une configuration optimale.
    Adapte automatiquement les paramètres selon les ressources disponibles.
    
    Returns:
        Dict: Configuration optimale pour l'environnement
    """
    try:
        import psutil
        
        # Récupérer les informations système
        cpu_count = psutil.cpu_count(logical=False)  # Cœurs physiques
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else None
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)
        
        # Configuration de base (pour système minimal)
        config = {
            'model_size': 'small',
            'n_threads': 2,
            'n_ctx': 1024,
            'n_batch': 256,
            'chunk_size': 300,
            'chunk_overlap': 50,
            'k': 3,
            'fetch_k': 5,
            'max_tokens': 100,
            'use_compression': False
        }
        
        # 1. Sélection du modèle selon la RAM disponible
        if memory_total_gb >= 16 and memory_available_gb >= 8:
            config['model_size'] = 'large'  # Mistral 7B
            config['max_tokens'] = 150
        elif memory_total_gb >= 8 and memory_available_gb >= 4:
            config['model_size'] = 'medium'  # Llama 3B
            config['max_tokens'] = 120
        else:
            config['model_size'] = 'small'  # Llama 1B
            config['max_tokens'] = 100
            
        # 2. Configuration CPU
        if cpu_count:
            if cpu_count >= 8 and (not cpu_freq or cpu_freq >= 2500):
                config['n_threads'] = min(8, cpu_count)
                config['n_ctx'] = 2048
                config['n_batch'] = 512
            elif cpu_count >= 4 and (not cpu_freq or cpu_freq >= 2000):
                config['n_threads'] = min(4, cpu_count)
                config['n_ctx'] = 1536
                config['n_batch'] = 384
            else:
                config['n_threads'] = 2
                config['n_ctx'] = 1024
                config['n_batch'] = 256
                
        # 3. Ajustements selon la charge système actuelle
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = memory.percent
        
        if cpu_percent > 80 or memory_percent > 80:
            # Réduire la charge si le système est déjà sous pression
            config['n_threads'] = max(1, config['n_threads'] - 1)
            config['n_ctx'] = max(512, config['n_ctx'] - 512)
            config['n_batch'] = max(128, config['n_batch'] - 128)
            config['chunk_size'] = 250
            config['k'] = 2
            config['fetch_k'] = 4
            
        # 4. Activation de la compression selon les ressources
        config['use_compression'] = (
            memory_total_gb >= 8 and 
            cpu_percent < 70 and 
            memory_percent < 70
        )
        
        # 5. Ajuster les paramètres de recherche selon la RAM
        if memory_available_gb < 2:
            config['k'] = 2
            config['fetch_k'] = 3
            config['chunk_size'] = 250
            config['chunk_overlap'] = 30
            
        # Loguer la configuration choisie
        logger.info(f"Configuration optimale détectée: {config}")
        logger.info(f"Ressources système - CPU: {cpu_count} cœurs, RAM: {memory_total_gb:.1f}GB")
        
        return config
        
    except Exception as e:
        logger.error(f"Erreur lors de la détection des ressources: {str(e)}")
        # Configuration minimale en cas d'erreur
        return {
            'model_size': 'small',
            'n_threads': 2,
            'n_ctx': 1024,
            'n_batch': 256,
            'chunk_size': 300,
            'chunk_overlap': 50,
            'k': 3,
            'fetch_k': 5,
            'max_tokens': 100,
            'use_compression': False
        }

def apply_optimal_config():
    """
    Applique la configuration optimale aux paramètres globaux.
    """
    try:
        optimal_config = get_optimal_config()
        
        # Mettre à jour MODEL_CONFIG
        config.MODEL_CONFIG.update({
            'model_path': str(config.MODELS_DIR / config.MODEL_CONFIGS[optimal_config['model_size']]['name']),
            'n_ctx': optimal_config['n_ctx'],
            'n_batch': optimal_config['n_batch'],
            'n_threads': optimal_config['n_threads'],
            'max_tokens': optimal_config['max_tokens']
        })
        
        # Mettre à jour VECTORDB_CONFIG
        config.VECTORDB_CONFIG.update({
            'chunk_size': optimal_config['chunk_size'],
            'chunk_overlap': optimal_config['chunk_overlap']
        })
        
        # Mettre à jour RETRIEVER_CONFIG
        config.RETRIEVER_CONFIG.update({
            'k': optimal_config['k'],
            'fetch_k': optimal_config['fetch_k'],
            'use_compression': optimal_config['use_compression']
        })
        
        logger.info("Configuration optimale appliquée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'application de la configuration optimale: {str(e)}")

def analyze_client_profile(client_info):
    """
    Analyse le profil du client pour déterminer les meilleures options de financement.
    
    Args:
        client_info: Dictionnaire contenant les informations du client
            {
                "budget_mensuel": float,
                "usage": str,
                "duree_souhaitee": int,
                "situation": str,
                "apport_possible": float,
                "km_annuel": int
            }
    
    Returns:
        Dict: Recommandations de financement
    """
    recommendations = {
        "solutions_recommandees": [],
        "vehicules_suggeres": [],
        "justification": "",
        "points_attention": []
    }
    
    # Charger les données des véhicules
    import json
    with open("documents/vehicules.json", "r", encoding="utf-8") as f:
        catalogue = json.load(f)
    
    # Déterminer la catégorie de véhicule adaptée au budget
    budget_total = client_info["budget_mensuel"] * client_info["duree_souhaitee"]
    categories_possibles = []
    for cat, info in catalogue["categories"].items():
        if info["budget_recommande"]["min"] <= budget_total <= info["budget_recommande"]["max"]:
            categories_possibles.append(cat)
    
    # Analyser l'usage pour déterminer le type de financement
    if client_info["usage"] == "professionnel":
        if client_info["km_annuel"] > 25000:
            recommendations["solutions_recommandees"].append({
                "type": "lld",
                "priorite": 1,
                "raison": "Usage professionnel intensif, solution fiscalement avantageuse"
            })
        else:
            recommendations["solutions_recommandees"].append({
                "type": "loa",
                "priorite": 1,
                "raison": "Usage professionnel modéré, flexibilité de renouvellement"
            })
    else:  # Usage personnel
        if client_info["apport_possible"] > 0:
            recommendations["solutions_recommandees"].append({
                "type": "credit",
                "priorite": 1,
                "raison": "Apport disponible, possibilité de devenir propriétaire"
            })
        else:
            recommendations["solutions_recommandees"].append({
                "type": "loa",
                "priorite": 1,
                "raison": "Pas d'apport significatif, mensualités optimisées"
            })
    
    # Suggérer des véhicules adaptés
    for categorie in categories_possibles:
        for vehicule in catalogue["vehicules"].get(categorie, []):
            # Vérifier si le véhicule correspond au budget mensuel
            for solution in recommendations["solutions_recommandees"]:
                option = vehicule["options_financement"][solution["type"]]
                if solution["type"] in ["loa", "lld"]:
                    if option["loyer"] <= client_info["budget_mensuel"]:
                        recommendations["vehicules_suggeres"].append({
                            "vehicule": f"{vehicule['marque']} {vehicule['modele']}",
                            "version": vehicule["version"],
                            "prix": vehicule["prix"],
                            "mensualite": option["loyer"],
                            "solution": solution["type"]
                        })
                elif solution["type"] == "credit":
                    # Calculer la mensualité approximative
                    taux_mensuel = option["taux"] / 1200  # Conversion en taux mensuel
                    montant_emprunte = vehicule["prix"] * (1 - option["apport_min"]/100)
                    mensualite = (montant_emprunte * taux_mensuel) / (1 - (1 + taux_mensuel)**(-client_info["duree_souhaitee"]))
                    if mensualite <= client_info["budget_mensuel"]:
                        recommendations["vehicules_suggeres"].append({
                            "vehicule": f"{vehicule['marque']} {vehicule['modele']}",
                            "version": vehicule["version"],
                            "prix": vehicule["prix"],
                            "mensualite": round(mensualite, 2),
                            "solution": "credit"
                        })
    
    # Ajouter des points d'attention
    if client_info["km_annuel"] > 25000:
        recommendations["points_attention"].append(
            "Kilométrage élevé : privilégier une formule avec entretien inclus"
        )
    if client_info["situation"] == "independant":
        recommendations["points_attention"].append(
            "Statut d'indépendant : prévoir les justificatifs de revenus sur 3 ans"
        )
    
    return recommendations

def format_recommendation_response(recommendations):
    """
    Formate les recommandations en réponse structurée.
    
    Args:
        recommendations: Dictionnaire des recommandations
        
    Returns:
        str: Réponse formatée
    """
    response = "📋 Analyse de votre profil et recommandations :\n\n"
    
    # Solutions recommandées
    response += "💡 Solutions de financement recommandées :\n"
    for solution in recommendations["solutions_recommandees"]:
        response += f"- {solution['type'].upper()} : {solution['raison']}\n"
    
    # Véhicules suggérés
    response += "\n🚗 Véhicules correspondant à vos critères :\n"
    for vehicule in recommendations["vehicules_suggeres"][:3]:  # Top 3
        response += f"- {vehicule['vehicule']} {vehicule['version']}\n"
        response += f"  Prix : {vehicule['prix']}€\n"
        response += f"  Mensualité estimée : {vehicule['mensualite']}€ en {vehicule['solution'].upper()}\n"
    
    # Points d'attention
    if recommendations["points_attention"]:
        response += "\n⚠️ Points d'attention :\n"
        for point in recommendations["points_attention"]:
            response += f"- {point}\n"
    
    return response

def get_carapi_jwt() -> str:
    """
    Récupère un JWT valide pour l'API CARAPI, soit depuis le cache soit en en générant un nouveau.
    
    Returns:
        str: Le JWT valide
    """
    try:
        # Vérifier si un JWT en cache existe et est valide
        if os.path.exists(CARAPI_CONFIG["jwt_cache_file"]):
            with open(CARAPI_CONFIG["jwt_cache_file"], "r") as f:
                cache_data = json.load(f)
                # Vérifier si le token n'est pas expiré
                if cache_data["exp"] > time.time():
                    return cache_data["token"]
                else:
                    print("JWT expiré, génération d'un nouveau token...")
        
        print("Tentative d'authentification CARAPI...")
        
        # Préparer les données d'authentification
        auth_data = {
            "api_token": CARAPI_CONFIG["api_token"],
            "api_secret": CARAPI_CONFIG["api_secret"]
        }
        
        # Générer un nouveau JWT
        response = requests.post(
            f"{CARAPI_CONFIG['base_url']}/auth/login",  # Endpoint correct selon la doc
            json=auth_data,
            headers={
                "accept": "text/plain",  # Header correct selon la doc
                "Content-Type": "application/json"
            }
        )
        
        print(f"URL appelée : {response.url}")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            # Le JWT est directement dans le corps de la réponse comme texte brut
            jwt = response.text.strip()
            
            if jwt and jwt.count('.') == 2:  # Vérification basique du format JWT
                try:
                    # Décoder le payload pour obtenir l'expiration
                    import base64
                    payload = jwt.split('.')[1]
                    # Ajouter le padding si nécessaire
                    padding = 4 - (len(payload) % 4)
                    if padding != 4:
                        payload += '=' * padding
                    
                    decoded_payload = json.loads(base64.b64decode(payload))
                    exp_timestamp = decoded_payload.get('exp')
                    
                    if not exp_timestamp:
                        print("Impossible de trouver l'expiration dans le JWT")
                        return None
                    
                    # Sauvegarder dans le cache
                    cache_data = {
                        "token": jwt,
                        "exp": exp_timestamp
                    }
                    
                    # Créer le dossier cache s'il n'existe pas
                    cache_dir = os.path.dirname(CARAPI_CONFIG["jwt_cache_file"])
                    os.makedirs(cache_dir, exist_ok=True)
                    
                    with open(CARAPI_CONFIG["jwt_cache_file"], "w") as f:
                        json.dump(cache_data, f)
                    
                    return jwt
                except Exception as e:
                    print(f"Erreur lors du décodage du JWT: {str(e)}")
                    return None
            else:
                print("Format de JWT invalide")
                return None
        else:
            print(f"Erreur d'authentification CARAPI: {response.text}")
            return None
            
    except Exception as e:
        print(f"Erreur lors de la récupération du JWT: {str(e)}")
        return None

def get_vehicle_data_from_api(filters: Optional[Dict] = None) -> Dict:
    """
    Récupère les données des véhicules depuis l'API CARAPI.
    
    Args:
        filters: Filtres optionnels pour la recherche (année, marque, modèle, etc.)
        
    Returns:
        Dict: Données des véhicules avec leurs spécifications
    """
    try:
        # Obtenir un JWT valide
        jwt = get_carapi_jwt()
        if not jwt:
            print("Impossible d'obtenir un JWT valide, utilisation des données locales")
            return load_vehicle_data()
        
        # Configuration des headers avec le JWT
        headers = {
            "Authorization": f"Bearer {jwt}",
            "accept": "application/json"
        }
        
        # Structure de base pour les données
        all_vehicles_data = {
            "categories": {
                "citadine": {
                    "description": "Véhicules compacts pour usage urbain",
                    "budget_recommande": {"min": 15000, "max": 25000}
                },
                "berline": {
                    "description": "Véhicules familiaux polyvalents",
                    "budget_recommande": {"min": 25000, "max": 45000}
                },
                "suv": {
                    "description": "Véhicules surélevés polyvalents",
                    "budget_recommande": {"min": 30000, "max": 60000}
                },
                "premium": {
                    "description": "Véhicules haut de gamme",
                    "budget_recommande": {"min": 50000, "max": 150000}
                }
            },
            "vehicules": {
                "citadine": [],
                "berline": [],
                "suv": [],
                "premium": []
            }
        }
        
        try:
            # 1. Obtenir la liste des marques
            makes_response = requests.get(
                f"{CARAPI_CONFIG['base_url']}/makes",
                headers=headers,
                timeout=10
            )
            
            print(f"Récupération des marques - Status: {makes_response.status_code}")
            
            if makes_response.status_code == 200:
                makes_data = makes_response.json()
                makes = makes_data.get("data", [])
                print(f"Nombre de marques trouvées: {len(makes)}")
                
                # 2. Pour chaque marque, obtenir les modèles récents
                total_vehicles_added = 0
                
                for make in makes:
                    make_name = make.get("name")
                    print(f"\nTraitement de la marque: {make_name}")
                    
                    # Construire les filtres pour les modèles récents
                    current_year = datetime.datetime.now().year
                    model_filters = [
                        {"field": "year", "op": ">=", "val": current_year - 2},  # Modèles des 2 dernières années
                        {"field": "make", "op": "=", "val": make_name}
                    ]
                    
                    # Récupérer les modèles de la marque
                    models_response = requests.get(
                        f"{CARAPI_CONFIG['base_url']}/models",
                        headers=headers,
                        params={
                            "json": json.dumps(model_filters),
                            "limit": 50,
                            "page": 1
                        },
                        timeout=10
                    )
                    
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        models = models_data.get("data", [])
                        print(f"Nombre de modèles trouvés pour {make_name}: {len(models)}")
                        
                        # 3. Pour chaque modèle, obtenir les versions (trims)
                        for model in models:
                            model_name = model.get("name")
                            model_id = model.get("id")
                            
                            if model_id:
                                trims_response = requests.get(
                                    f"{CARAPI_CONFIG['base_url']}/models/{model_id}/trims",
                                    headers=headers,
                                    timeout=10
                                )
                                
                                if trims_response.status_code == 200:
                                    trims_data = trims_response.json()
                                    trims = trims_data.get("data", [])
                                    print(f"  - {model_name}: {len(trims)} versions trouvées")
                                    
                                    for trim in trims:
                                        try:
                                            # Déterminer la catégorie du véhicule
                                            category = determine_vehicle_category(trim)
                                            
                                            # Créer l'entrée du véhicule
                                            vehicle_entry = {
                                                "marque": make_name,
                                                "modele": model_name,
                                                "version": trim.get("name", "Standard"),
                                                "prix": trim.get("msrp", 0),
                                                "options_financement": calculate_financing_options(trim)
                                            }
                                            
                                            # Vérifier si le prix est valide
                                            if vehicle_entry["prix"] > 0:
                                                # Ajouter le véhicule à sa catégorie
                                                all_vehicles_data["vehicules"][category].append(vehicle_entry)
                                                total_vehicles_added += 1
                                                print(f"    ✓ Ajout: {make_name} {model_name} {vehicle_entry['version']} ({category})")
                                            else:
                                                print(f"    ✗ Prix non valide pour: {make_name} {model_name} {vehicle_entry['version']}")
                                                
                                        except Exception as e:
                                            print(f"    ✗ Erreur lors du traitement de la version: {str(e)}")
                                            continue
                                else:
                                    print(f"  ✗ Erreur lors de la récupération des versions pour {model_name}: {trims_response.status_code}")
                
                print(f"\nTotal des véhicules ajoutés: {total_vehicles_added}")
                
                # Sauvegarder les données
                save_vehicle_data(all_vehicles_data)
                return all_vehicles_data
                
            else:
                print(f"Erreur lors de la récupération des marques: {makes_response.text}")
                return load_vehicle_data()
                
        except RequestException as re:
            print(f"Erreur de requête API: {str(re)}")
            return load_vehicle_data()
            
    except Exception as e:
        print(f"Erreur lors de la récupération des données: {str(e)}")
        return load_vehicle_data()

def determine_vehicle_category(vehicle: Dict) -> str:
    """
    Détermine la catégorie d'un véhicule en fonction de ses caractéristiques.
    """
    try:
        # Récupérer le prix de base
        price = vehicle.get("msrp", 0)
        
        # Récupérer le type de carrosserie
        body_style = vehicle.get("body_style", "").lower()
        body_type = vehicle.get("body_type", "").lower()
        
        # Mots-clés pour identifier les SUVs
        suv_keywords = ["suv", "crossover", "cuv", "4x4", "crossover utility vehicle"]
        
        # Mots-clés pour identifier les berlines
        berline_keywords = ["sedan", "berline", "saloon"]
        
        # Mots-clés pour identifier les citadines
        citadine_keywords = ["hatchback", "compact", "city car", "supermini"]
        
        # Déterminer la catégorie en fonction du type et du prix
        if any(keyword in body_style or keyword in body_type for keyword in suv_keywords):
            if price >= 60000:
                return "premium"
            return "suv"
        elif price >= 50000:
            return "premium"
        elif any(keyword in body_style or keyword in body_type for keyword in berline_keywords):
            return "berline"
        elif any(keyword in body_style or keyword in body_type for keyword in citadine_keywords) or price < 25000:
            return "citadine"
        else:
            return "berline"  # Par défaut
            
    except Exception as e:
        print(f"Erreur lors de la détermination de la catégorie: {str(e)}")
        return "berline"  # Catégorie par défaut en cas d'erreur

def calculate_financing_options(vehicle: Dict) -> Dict:
    """
    Calcule les options de financement pour un véhicule.
    """
    try:
        # Récupérer le prix de base
        price = vehicle.get("msrp", 0)
        if not price or price <= 0:
            price = 30000  # Prix par défaut si non disponible
        
        # Calculer les différentes options de financement
        return {
            "comptant": {
                "apport_minimum": round(price * 0.1),  # 10% d'apport minimum
                "mensualite": 0,
                "duree": 0
            },
            "credit": {
                "apport_minimum": round(price * 0.1),  # 10% d'apport minimum
                "mensualite": round((price * 0.9) / 60),  # Estimation sur 60 mois
                "duree": 60
            },
            "loa": {
                "apport_minimum": round(price * 0.1),  # 10% d'apport minimum
                "mensualite": round(price * 0.015),  # Environ 1.5% du prix en mensualité
                "duree": 48
            }
        }
    except Exception as e:
        print(f"Erreur lors du calcul des options de financement: {str(e)}")
        return {
            "comptant": {"apport_minimum": 0, "mensualite": 0, "duree": 0},
            "credit": {"apport_minimum": 0, "mensualite": 0, "duree": 60},
            "loa": {"apport_minimum": 0, "mensualite": 0, "duree": 48}
        }

def save_vehicle_data(data: Dict) -> None:
    """
    Sauvegarde les données des véhicules dans un fichier local.
    """
    try:
        # Créer le dossier documents s'il n'existe pas
        os.makedirs("documents", exist_ok=True)
        
        # Vérifier si nous avons des données à sauvegarder
        if not data or not isinstance(data, dict):
            print("Aucune donnée valide à sauvegarder")
            return
            
        vehicles_count = sum(len(vehicles) for vehicles in data.get("vehicules", {}).values())
        print(f"Sauvegarde de {vehicles_count} véhicules dans {len(data.get('vehicules', {}))} catégories")
        
        # Sauvegarder les données
        with open("documents/vehicules.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        print("Données sauvegardées avec succès dans vehicules.json")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des données: {str(e)}")
        import traceback
        print(traceback.format_exc())

def load_vehicle_data() -> Dict:
    """
    Charge les données des véhicules depuis le fichier local.
    """
    try:
        with open("documents/vehicules.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement des données locales: {str(e)}")
        return {}

def update_vehicle_database(frequency: str = "daily") -> None:
    """
    Met à jour la base de données des véhicules selon la fréquence spécifiée.
    
    Args:
        frequency: Fréquence de mise à jour ("daily", "weekly", "monthly")
    """
    try:
        # Vérifier la dernière mise à jour
        last_update = None
        try:
            with open("documents/last_update.txt", "r") as f:
                last_update = datetime.datetime.fromisoformat(f.read().strip())
        except:
            last_update = None
        
        current_time = datetime.datetime.now()
        
        # Déterminer si une mise à jour est nécessaire
        update_needed = False
        if not last_update:
            update_needed = True
        elif frequency == "daily" and (current_time - last_update).days >= 1:
            update_needed = True
        elif frequency == "weekly" and (current_time - last_update).days >= 7:
            update_needed = True
        elif frequency == "monthly" and (current_time - last_update).days >= 30:
            update_needed = True
            
        if update_needed:
            # Mettre à jour les données
            get_vehicle_data_from_api()
            
            # Enregistrer la date de mise à jour
            with open("documents/last_update.txt", "w") as f:
                f.write(current_time.isoformat())
                
            print("Base de données des véhicules mise à jour avec succès.")
        else:
            print("La base de données des véhicules est à jour.")
            
    except Exception as e:
        print(f"Erreur lors de la mise à jour de la base de données: {str(e)}") 

# === Fonction de détection de langue intelligente ===
def detect_language_smart(text):
    """
    Détecte la langue d'un texte de façon intelligente.
    
    Args:
        text: Le texte à analyser
        
    Returns:
        Code de langue détecté ou 'fr' par défaut
    """
    if not text or len(text.strip()) < 3:
        return 'fr'  # Défaut français pour textes trop courts
    
    text = text.strip().lower()
    
    # Détection par mots-clés communs pour éviter les erreurs de langdetect
    keyword_patterns = {
        'fr': ['bonjour', 'salut', 'merci', 'oui', 'non', 'comment', 'pourquoi', 'quoi', 'qui', 'où', 'quand', 
               'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'le', 'la', 'les', 'un', 'une', 'des',
               'est', 'sont', 'avoir', 'être', 'faire', 'aller', 'venir', 'voir', 'savoir', 'pouvoir',
               'crédit', 'prêt', 'financement', 'voiture', 'automobile', 'ça', 'marche', 'veux', 'acheter'],
        'en': ['hello', 'hi', 'thanks', 'thank', 'yes', 'no', 'how', 'why', 'what', 'who', 'where', 'when',
               'i', 'you', 'he', 'she', 'we', 'they', 'the', 'a', 'an', 'and', 'or', 'but',
               'is', 'are', 'have', 'has', 'be', 'do', 'can', 'will', 'would', 'could',
               'credit', 'loan', 'financing', 'car', 'auto', 'vehicle', 'does', 'work', 'want', 'buy'],
        'es': ['hola', 'gracias', 'sí', 'no', 'cómo', 'por qué', 'qué', 'quién', 'dónde', 'cuándo',
               'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'el', 'la', 'los', 'las',
               'es', 'son', 'tener', 'ser', 'hacer', 'ir', 'venir', 'ver', 'saber', 'poder',
               'crédito', 'préstamo', 'financiación', 'coche', 'auto', 'funciona', 'quiero', 'comprar', 'diferencia'],
        'de': ['hallo', 'danke', 'ja', 'nein', 'wie', 'warum', 'was', 'wer', 'wo', 'wann',
               'ich', 'du', 'er', 'sie', 'wir', 'ihr', 'der', 'die', 'das', 'ein', 'eine',
               'ist', 'sind', 'haben', 'sein', 'machen', 'gehen', 'kommen', 'sehen', 'wissen', 'können',
               'kredit', 'darlehen', 'finanzierung', 'auto', 'wagen', 'funktioniert', 'möchte', 'kaufen', 'unterschied'],
        'it': ['ciao', 'grazie', 'sì', 'no', 'come', 'perché', 'cosa', 'chi', 'dove', 'quando',
               'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro', 'il', 'la', 'lo', 'gli', 'le',
               'è', 'sono', 'avere', 'essere', 'fare', 'andare', 'venire', 'vedere', 'sapere', 'potere',
               'credito', 'prestito', 'finanziamento', 'auto', 'macchina', 'funziona', 'voglio', 'comprare', 'differenza'],
        'ar': ['مرحبا', 'شكرا', 'نعم', 'لا', 'كيف', 'لماذا', 'ماذا', 'من', 'أين', 'متى',
               'أنا', 'أنت', 'هو', 'هي', 'نحن', 'أنتم', 'هم', 'هن', 'ال', 'في', 'على',
               'قرض', 'تمويل', 'سيارة', 'مركبة', 'يعمل', 'أريد', 'شراء', 'الفرق']
    }
    
    # Compter les correspondances pour chaque langue
    scores = {}
    for lang, keywords in keyword_patterns.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        scores[lang] = score
    
    # Si on a des correspondances claires par mots-clés
    max_score = max(scores.values())
    if max_score > 0:
        detected_lang = max(scores, key=scores.get)
        
        # Vérifier si la différence est significative (au moins 2 points d'écart pour les cas ambigus)
        second_highest = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        if max_score - second_highest >= 1:  # Au moins 1 point d'écart
            return detected_lang
    
    # Sinon, utiliser langdetect avec gestion d'erreurs
    try:
        import langdetect
        detected = langdetect.detect(text)
        
        # Mapping des codes de langue
        lang_mapping = {
            'fr': 'fr',
            'en': 'en', 
            'es': 'es',
            'de': 'de',
            'it': 'it',
            'ar': 'ar',
            'ca': 'es',  # Catalan -> Espagnol
            'pt': 'es',  # Portugais -> Espagnol  
            'nl': 'de',  # Néerlandais -> Allemand
            'fi': 'en',  # Finnois -> Anglais (pour éviter l'erreur "salut" -> finnois)
            'sv': 'en',  # Suédois -> Anglais
            'no': 'en',  # Norvégien -> Anglais
            'da': 'en',  # Danois -> Anglais
        }
        
        return lang_mapping.get(detected, 'fr')  # Défaut français
        
    except:
        # Si tout échoue, retourner la langue avec le plus de correspondances ou français
        if max_score > 0:
            return max(scores, key=scores.get)
        return 'fr'  # Défaut français en cas d'erreur

def get_language_message(lang_code, message_type):
    """
    Récupère un message dans la langue appropriée.
    
    Args:
        lang_code: Code de langue
        message_type: Type de message ('greeting', 'no_info', 'error', 'fallback')
        
    Returns:
        Message dans la langue demandée
    """
    import config
    
    # Vérifier si la langue est supportée
    if lang_code in config.LANGUAGE_MESSAGES:
        return config.LANGUAGE_MESSAGES[lang_code].get(message_type, 
                                                      config.LANGUAGE_MESSAGES['fr'][message_type])
    else:
        # Langue non supportée, utiliser l'anglais
        return config.LANGUAGE_MESSAGES['en'].get(message_type, 
                                                 config.LANGUAGE_MESSAGES['fr'][message_type]) 

def validate_response(response, context, question):
    """
    Valide la réponse du modèle avec des contrôles stricts.
    
    Args:
        response: Réponse générée
        context: Contexte utilisé
        question: Question posée
        
    Returns:
        tuple: (réponse validée, score de confiance)
    """
    if not response or not context:
        return get_fallback_response(question), 0.0
    
    # 1. Vérification de la longueur
    if len(response.split()) > 50:  # Trop long
        sentences = response.split('.')
        response = '. '.join(s.strip() for s in sentences[:2] if s.strip()) + '.'
    
    # 2. Extraction des mots-clés du contexte et de la réponse
    context_words = set(w.lower() for w in context.split() if len(w) > 3)
    response_words = set(w.lower() for w in response.split() if len(w) > 3)
    
    # 3. Calcul du chevauchement
    overlap_words = context_words.intersection(response_words)
    overlap_score = len(overlap_words) / len(response_words) if response_words else 0
    
    # 4. Vérification des termes suspects
    suspect_phrases = config.VALIDATION_CONFIG['banned_phrases']
    contains_suspect = any(phrase in response.lower() for phrase in suspect_phrases)
    
    # 5. Calcul du score de confiance
    confidence_score = 1.0
    
    # Pénalités
    if contains_suspect:
        confidence_score *= 0.5
    if overlap_score < 0.3:  # Au moins 30% des mots doivent venir du contexte
        confidence_score *= 0.3
    if len(overlap_words) < config.VALIDATION_CONFIG['min_context_overlap']:
        confidence_score *= 0.4
        
    # 6. Vérification des critères de rejet
    should_reject = (
        len(overlap_words) == 0 or  # Aucun mot-clé du contexte
        confidence_score < config.VALIDATION_CONFIG['min_confidence'] or  # Confiance trop faible
        contains_suspect  # Contient des phrases suspectes
    )
    
    if should_reject:
        return get_fallback_response(question), 0.0
        
    return response.strip(), confidence_score

def get_fallback_response(question):
    """
    Génère une réponse de repli appropriée basée sur la question.
    
    Args:
        question: Question posée
        
    Returns:
        str: Réponse de repli
    """
    # Charger les produits financiers
    try:
        with open("documents/produits_financiers.json", "r", encoding="utf-8") as f:
            produits = json.load(f)
            
        # Chercher si la question concerne un produit spécifique
        for produit_id, produit in produits["produits"].items():
            if any(mot in question.lower() for mot in produit["nom"].lower().split()):
                return f"Pour le {produit['nom']}, je vous invite à consulter directement nos conseillers pour des informations précises et personnalisées."
    except:
        pass
    
    # Réponse générique si aucun produit spécifique n'est trouvé
    return "Je ne peux pas fournir une réponse précise à cette question. Pour obtenir des informations fiables, je vous invite à reformuler votre question ou à contacter directement nos conseillers."

def format_response(response: str, confidence: float) -> str:
    """Formate la réponse en ajoutant une indication de confiance."""
    try:
        confidence_pct = int(confidence * 100)
        return f"{response}\n\nConfiance: {confidence_pct}%"
    except Exception:
        return response

def timeout(seconds):
    """
    Décorateur pour ajouter un timeout aux fonctions.
    Utilise signal.SIGALRM sur Unix et une approche thread sur Windows.
    
    Args:
        seconds: Nombre de secondes avant timeout
    """
    def decorator(func):
        if os.name == 'nt':  # Windows
            import threading
            import _thread
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = []
                def worker():
                    try:
                        result.append(func(*args, **kwargs))
                    except Exception as e:
                        result.append(e)
                
                thread = threading.Thread(target=worker)
                thread.daemon = True
                thread.start()
                thread.join(seconds)
                
                if thread.is_alive():
                    _thread.interrupt_main()  # Force une interruption
                    raise TimeoutError(f"Fonction {func.__name__} a dépassé le délai de {seconds} secondes")
                
                if result and isinstance(result[0], Exception):
                    raise result[0]
                return result[0] if result else None
                
        else:  # Unix/Linux
            @wraps(func)
            def wrapper(*args, **kwargs):
                def handler(signum, frame):
                    raise TimeoutError(f"Fonction {func.__name__} a dépassé le délai de {seconds} secondes")
                
                # Configurer le signal
                old_handler = signal.signal(signal.SIGALRM, handler)
                signal.alarm(seconds)
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    # Restaurer le gestionnaire précédent
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                
                return result
                
        return wrapper
    return decorator

@timeout(config.TIMEOUT_CONFIG['query'])
def call_model_with_timeout(prompt):
    """
    Appelle le modèle avec un timeout.
    """
    llm = load_llm()
    if not llm:
        raise Exception("Modèle non disponible")
    return llm(prompt)

def generate_response(question, context, theme=None):
    """
    Génère une réponse en utilisant strictement le contexte fourni.
    """
    if not context:
        return get_fallback_response(question)
        
    # Construire le prompt avec instructions strictes
    prompt = f"""{config.SYSTEM_PROMPT}

CONTEXTE STRICT:
{context}

QUESTION: {question}

RÈGLES ABSOLUES:
1. Utilise UNIQUEMENT les informations du contexte ci-dessus
2. Si l'information n'est pas dans le contexte, réponds UNIQUEMENT: "Je n'ai pas cette information dans mes données actuelles."
3. Ne fais JAMAIS de suppositions ou de généralisations
4. Limite TOUJOURS tes réponses à 2 phrases maximum
5. N'ajoute AUCUN commentaire personnel
6. N'utilise JAMAIS de termes vagues ou incertains
7. Ne donne JAMAIS d'informations non vérifiées

RÉPONSE:"""
    
    try:
        # Appeler le modèle avec timeout
        response = call_model_with_timeout(prompt)
        return response
    except TimeoutError:
        logger.error("Timeout lors de la génération de la réponse")
        return get_fallback_response(question)
    except Exception as e:
        logger.error(f"Erreur de génération: {str(e)}")
        return get_fallback_response(question)

def process_user_message(user_message, theme=None):
    """
    Traite le message de l'utilisateur avec validation et formatage améliorés.
    """
    try:
        # Nettoyage et validation de base
        cleaned_message = clean_user_input(user_message)
        if not cleaned_message:
            return "Je n'ai pas compris votre message. Pouvez-vous le reformuler ?", [], {}
        
        # Vérifier si c'est une salutation
        if is_greeting(cleaned_message):
            return get_greeting_response(), [], {"type": "greeting", "confiance": 1.0}
        
        # Vérifier si c'est une question sur un produit spécifique
        with open("documents/produits_financiers.json", "r", encoding="utf-8") as f:
            produits = json.load(f)
            
        for produit_id, produit in produits["produits"].items():
            if any(keyword in cleaned_message.lower() for keyword in produit["nom"].lower().split()):
                # Question spécifique sur un produit
                if "avantage" in cleaned_message.lower():
                    return "\n".join(produit["avantages"]), [], {"source": "base_produits", "confiance": 1.0}
                elif "condition" in cleaned_message.lower():
                    conditions = [f"{k}: {v}" for k, v in produit["conditions"].items()]
                    return "\n".join(conditions), [], {"source": "base_produits", "confiance": 1.0}
                else:
                    return f"{produit['description']}\n\nIdéal pour :\n" + "\n".join(produit["ideal_pour"]), [], {"source": "base_produits", "confiance": 1.0}
        
        # Si ce n'est pas une question spécifique sur un produit, utiliser le
        # traitement normal. La fonction generate_response renvoie uniquement la
        # réponse, on initialise donc une liste vide pour les sources afin de
        # maintenir la compatibilité de la sortie.
        response = generate_response(cleaned_message, theme)
        sources: List = []
        validated_response, confidence = validate_response(
            response,
            " ".join(sources),
            cleaned_message,
        )
        formatted_response = format_response(validated_response, confidence)

        return formatted_response, sources, {
            "confiance": confidence,
            "temps_reponse": measure_response_time(),
            "theme": theme,
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du message: {str(e)}")
        return "Je rencontre des difficultés techniques. Pouvez-vous reformuler votre question ?", [], {"error": str(e)} 

def is_greeting(text: str) -> bool:
    """
    Vérifie si le texte est une salutation.
    
    Args:
        text: Texte à vérifier
        
    Returns:
        bool: True si c'est une salutation
    """
    greetings = {
        'fr': ['bonjour', 'salut', 'bonsoir', 'coucou', 'hello', 'hey'],
        'en': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'],
        'es': ['hola', 'buenos dias', 'buenas tardes', 'buenas noches'],
        'de': ['hallo', 'guten tag', 'guten morgen', 'guten abend'],
        'it': ['ciao', 'buongiorno', 'buonasera', 'salve']
    }
    
    text = text.lower().strip()
    
    # Vérifier dans toutes les langues
    for lang_greetings in greetings.values():
        if any(text.startswith(greeting) for greeting in lang_greetings):
            return True
            
    return False

def get_greeting_response() -> str:
    """
    Génère une réponse de salutation appropriée.
    
    Returns:
        str: Message de salutation
    """
    # Détecter l'heure pour adapter la salutation
    hour = datetime.datetime.now().hour
    
    if hour < 12:
        time_greeting = "Bonjour"
    elif hour < 18:
        time_greeting = "Bonjour"
    else:
        time_greeting = "Bonsoir"
        
    return f"{time_greeting} ! Je suis votre assistant financier TeamWill. Comment puis-je vous aider aujourd'hui ?" 

def clean_user_input(text: str) -> str:
    """
    Nettoie et normalise le texte d'entrée de l'utilisateur.
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        str: Texte nettoyé
    """
    if not text:
        return ""
        
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer les espaces multiples
    text = " ".join(text.split())
    
    # Supprimer les caractères spéciaux sauf la ponctuation de base
    import re
    text = re.sub(r'[^\w\s.,!?€$-]', '', text)
    
    # Normaliser les caractères accentués
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    
    return text.strip() 