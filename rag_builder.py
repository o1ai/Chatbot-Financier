"""
Indexation des documents pour le chatbot TeamWill.
Ce script permet de créer ou mettre à jour la base vectorielle pour le RAG.
"""

import argparse
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import sys
import os

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_core.documents import Document

import config
import utils

def detect_language(text: str) -> str:
    """
    Détecte la langue du texte.
    
    Args:
        text: Le texte à analyser
        
    Returns:
        Le code de langue détecté (fr, en, etc.)
    """
    try:
        from langdetect import detect
        return detect(text)
    except:
        # Fallback simple
        fr_words = ["et", "le", "la", "les", "un", "une", "des", "est", "sont"]
        en_words = ["the", "and", "is", "are", "a", "an", "of", "to", "in"]
        
        text_lower = text.lower()
        fr_count = sum(1 for word in fr_words if f" {word} " in text_lower)
        en_count = sum(1 for word in en_words if f" {word} " in text_lower)
        
        return "fr" if fr_count > en_count else "en"

def extract_document_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Extrait les métadonnées d'un document.
    
    Args:
        file_path: Chemin vers le document
        
    Returns:
        Un dictionnaire contenant les métadonnées
    """
    metadata = {
        "source": file_path.name,
        "file_type": file_path.suffix.lower().replace(".", ""),
        "file_size": file_path.stat().st_size,
        "creation_date": time.ctime(file_path.stat().st_ctime),
        "modification_date": time.ctime(file_path.stat().st_mtime),
    }
    
    # Pour les PDFs, essayer d'extraire des métadonnées supplémentaires
    if file_path.suffix.lower() == ".pdf":
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                info = pdf.metadata
                if info:
                    for key, value in info.items():
                        if key.startswith('/'):
                            key = key[1:]
                        if value and isinstance(value, str):
                            metadata[key] = value
        except:
            pass
    
    return metadata

def load_document(file_path: Path) -> List[Document]:
    """
    Charge un document et retourne une liste de Documents LangChain.
    
    Args:
        file_path: Chemin vers le document à charger
        
    Returns:
        Liste de Documents LangChain
    """
    print(f"🔄 Chargement de {file_path.name}...")
    
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_extension in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(str(file_path))
        elif file_extension == ".csv":
            loader = CSVLoader(str(file_path))
        elif file_extension == ".json":
            loader = JSONLoader(
                file_path=str(file_path),
                jq_schema=".",
                text_content=False
            )
        else:
            # Fallback pour les fichiers texte
            loader = TextLoader(str(file_path))
        
        docs = loader.load()
        
        # Ajouter des métadonnées avancées
        doc_metadata = extract_document_metadata(file_path)
        
        # Détection de langue sur un échantillon du texte
        if docs:
            sample_text = docs[0].page_content[:1000]
            language = detect_language(sample_text)
            doc_metadata["language"] = language
        
        # Enrichir chaque document avec les métadonnées
        for doc in docs:
            doc.metadata.update(doc_metadata)
        
        print(f"✓ {file_path.name} chargé avec succès: {len(docs)} pages/sections")
        return docs
    
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {file_path.name}: {str(e)}")
        return []

def load_all_documents(directory: Path = None, include_patterns: List[str] = None) -> List[Document]:
    """
    Charge tous les documents du répertoire spécifié.
    
    Args:
        directory: Répertoire contenant les documents (défaut: config.DOCUMENTS_DIR)
        include_patterns: Liste de patterns de noms de fichiers à inclure
        
    Returns:
        Liste de tous les Documents chargés
    """
    directory = directory or config.DOCUMENTS_DIR
    include_patterns = include_patterns or ["*.pdf", "*.docx", "*.doc", "*.txt", "*.csv", "*.json"]
    
    print(f"🔍 Recherche de documents dans {directory}...")
    
    # Collecter tous les fichiers correspondant aux patterns
    all_files = []
    for pattern in include_patterns:
        all_files.extend(list(directory.glob(pattern)))
    
    if not all_files:
        print(f"⚠️ Aucun document trouvé dans {directory} avec les patterns {include_patterns}")
        return []
    
    print(f"📚 {len(all_files)} document(s) trouvé(s)")
    
    # Charger tous les documents avec une barre de progression
    all_documents = []
    for file in tqdm(all_files, desc="Chargement des documents", unit="file"):
        docs = load_document(file)
        all_documents.extend(docs)
    
    print(f"✓ {len(all_documents)} fragments chargés au total")
    return all_documents

def split_documents(documents: List[Document], config_override: Dict[str, Any] = None) -> List[Document]:
    """
    Découpe les documents en chunks plus petits.
    
    Args:
        documents: Liste de Documents à découper
        config_override: Paramètres de découpage personnalisés
        
    Returns:
        Liste de Documents découpés
    """
    if not documents:
        return []
    
    # Créer la configuration de découpage
    splitter_config = dict(config.TEXT_SPLITTER_CONFIG)
    if config_override:
        splitter_config.update(config_override)
    
    print(f"✂️ Découpage des documents avec chunk_size={splitter_config['chunk_size']}, overlap={splitter_config['chunk_overlap']}")
    
    # Créer le text splitter
    text_splitter = RecursiveCharacterTextSplitter(**splitter_config)
    
    # Découper les documents
    chunks = text_splitter.split_documents(documents)
    
    print(f"✓ {len(chunks)} fragments créés à partir de {len(documents)} documents/sections")
    return chunks

def create_vector_db(documents: List[Document], rebuild: bool = False) -> Chroma:
    """
    Crée ou met à jour la base vectorielle.
    
    Args:
        documents: Liste de Documents à ajouter à la base
        rebuild: Si True, reconstruit entièrement la base (supprime l'existante)
        
    Returns:
        Instance de la base vectorielle Chroma
    """
    if not documents:
        print("⚠️ Aucun document à indexer")
        return None
    
    # Obtenir l'embedding
    embedding = utils.get_embeddings()
    
    # Chemin de la base vectorielle
    vectordb_dir = config.VECTORDB_DIR
    
    # Reconstruire ou mettre à jour
    if rebuild and vectordb_dir.exists():
        import shutil
        print(f"🗑️ Suppression de la base vectorielle existante: {vectordb_dir}")
        shutil.rmtree(vectordb_dir)
    
    # Création/mise à jour de la base
    print(f"💾 {'Création' if rebuild else 'Mise à jour'} de la base vectorielle...")
    start_time = time.time()
    
    if rebuild or not vectordb_dir.exists():
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=str(vectordb_dir)
        )
    else:
        # Chargement de la base existante
        vectordb = Chroma(
            persist_directory=str(vectordb_dir),
            embedding_function=embedding
        )
        # Ajout des nouveaux documents
        vectordb.add_documents(documents)
    
    # Persistance
    vectordb.persist()
    
    end_time = time.time()
    print(f"✓ Base vectorielle {'créée' if rebuild else 'mise à jour'} avec succès: {len(documents)} fragments indexés")
    print(f"⏱️ Temps d'indexation: {end_time - start_time:.2f}s")
    
    return vectordb

def export_document_summary(documents: List[Document], output_file: str = None):
    """
    Exporte un résumé des documents indexés.
    
    Args:
        documents: Liste de Documents indexés
        output_file: Chemin du fichier de sortie (défaut: documents_summary.csv)
    """
    if not documents:
        return
    
    output_file = output_file or "documents_summary.csv"
    
    # Créer un DataFrame avec les informations sur les documents
    data = []
    for doc in documents:
        entry = {
            "source": doc.metadata.get("source", "Unknown"),
            "language": doc.metadata.get("language", "Unknown"),
            "chunk_size": len(doc.page_content),
            "file_type": doc.metadata.get("file_type", "Unknown"),
            "creation_date": doc.metadata.get("creation_date", "Unknown")
        }
        data.append(entry)
    
    df = pd.DataFrame(data)
    
    # Créer un résumé
    summary = df.groupby(["source", "language", "file_type"]).agg(
        chunk_count=("chunk_size", "count"),
        avg_chunk_size=("chunk_size", "mean"),
        total_characters=("chunk_size", "sum")
    ).reset_index()
    
    # Exporter le résumé
    summary.to_csv(output_file, index=False)
    print(f"📊 Résumé des documents exporté vers {output_file}")

def main():
    """Fonction principale pour l'indexation des documents."""
    # Analyse des arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Indexation des documents pour le chatbot TeamWill")
    parser.add_argument("--rebuild", action="store_true", help="Reconstruire entièrement la base vectorielle")
    parser.add_argument("--dir", type=str, help="Répertoire contenant les documents")
    parser.add_argument("--chunk-size", type=int, help="Taille des fragments")
    parser.add_argument("--chunk-overlap", type=int, help="Chevauchement des fragments")
    parser.add_argument("--include", nargs="+", help="Patterns de fichiers à inclure")
    parser.add_argument("--summary", action="store_true", help="Générer un résumé des documents indexés")
    args = parser.parse_args()
    
    # Configuration personnalisée pour le text splitter
    splitter_config = {}
    if args.chunk_size:
        splitter_config["chunk_size"] = args.chunk_size
    if args.chunk_overlap:
        splitter_config["chunk_overlap"] = args.chunk_overlap
    
    # Répertoire des documents
    documents_dir = Path(args.dir) if args.dir else config.DOCUMENTS_DIR
    
    # Patterns d'inclusion
    include_patterns = args.include if args.include else None
    
    # Chargement des documents
    all_documents = load_all_documents(
        directory=documents_dir,
        include_patterns=include_patterns
    )
    
    if not all_documents:
        print("❌ Aucun document chargé. Vérifiez le répertoire et les patterns d'inclusion.")
        sys.exit(1)
    
    # Découpage des documents
    chunks = split_documents(all_documents, splitter_config)
    
    # Création/mise à jour de la base vectorielle
    db = create_vector_db(chunks, rebuild=args.rebuild)
    
    # Génération du résumé si demandé
    if args.summary:
        export_document_summary(chunks)
    
    print("\n=== Indexation terminée. Vous pouvez maintenant lancer chatbot.py ===")

if __name__ == "__main__":
    main() 