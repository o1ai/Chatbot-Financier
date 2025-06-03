"""
Script de diagnostic pour identifier les problèmes du chatbot RAG
"""

import json
from pathlib import Path
import sys
sys.path.append('.')

import config
import utils

def test_vectordb():
    """Test de la base vectorielle"""
    print("🔍 Test de la base vectorielle...")
    
    try:
        vectordb = utils.get_vectordb()
        if not vectordb:
            print("❌ Impossible de charger la base vectorielle")
            return False
        
        # Vérifier le nombre de documents
        collection_count = vectordb._collection.count()
        print(f"📊 Nombre de documents dans la base: {collection_count}")
        
        if collection_count == 0:
            print("❌ La base vectorielle est vide !")
            return False
        
        # Test de recherche simple
        test_query = "crédit automobile"
        results = vectordb.similarity_search(test_query, k=3)
        print(f"🔍 Test de recherche pour '{test_query}': {len(results)} résultats")
        
        if results:
            print(f"✅ Premier résultat (extrait): {results[0].page_content[:200]}...")
            return True
        else:
            print("❌ Aucun résultat de recherche")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test de la base vectorielle: {str(e)}")
        return False

def test_retriever():
    """Test du retriever"""
    print("\n🔍 Test du retriever...")
    
    try:
        retriever = utils.get_retriever(theme="Tous", use_compression=False)
        if not retriever:
            print("❌ Impossible de créer le retriever")
            return False
        
        # Test de récupération
        test_query = "Qu'est-ce que le crédit automobile ?"
        docs = retriever.get_relevant_documents(test_query)
        print(f"📝 Requête: '{test_query}'")
        print(f"📊 Documents récupérés: {len(docs)}")
        
        if docs:
            for i, doc in enumerate(docs):
                print(f"📄 Document {i+1}: {doc.page_content[:150]}...")
                print(f"   Source: {doc.metadata.get('source', 'Inconnue')}")
            return True
        else:
            print("❌ Aucun document récupéré")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test du retriever: {str(e)}")
        return False

def test_llm():
    """Test du modèle LLM"""
    print("\n🔍 Test du modèle LLM...")
    
    try:
        from chatbot import load_llm
        llm = load_llm()
        
        if not llm:
            print("❌ Impossible de charger le modèle LLM")
            return False
        
        # Test simple
        test_prompt = "Réponds en une phrase: qu'est-ce qu'un crédit ?"
        response = llm(test_prompt)
        print(f"🤖 Test LLM - Réponse: {response}")
        
        if response and len(response.strip()) > 10:
            print("✅ Le modèle LLM fonctionne")
            return True
        else:
            print("❌ Réponse LLM invalide")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test du LLM: {str(e)}")
        return False

def test_full_chain():
    """Test de la chaîne complète RAG"""
    print("\n🔍 Test de la chaîne RAG complète...")
    
    try:
        from chatbot import get_qa_chain
        
        # Créer le retriever
        retriever = utils.get_retriever(theme="Tous", use_compression=False)
        if not retriever:
            print("❌ Échec du retriever")
            return False
        
        # Créer la chaîne QA
        qa_chain = get_qa_chain(retriever, detected_lang="fr")
        if not qa_chain:
            print("❌ Échec de création de la chaîne QA")
            return False
        
        # Test de la chaîne
        test_question = "Comment fonctionne un crédit automobile ?"
        print(f"❓ Question test: '{test_question}'")
        
        response = qa_chain({"question": test_question})
        
        answer = response.get("answer", "")
        sources = response.get("source_documents", [])
        
        print(f"🤖 Réponse: {answer}")
        print(f"📄 Nombre de sources: {len(sources)}")
        
        if sources:
            for i, doc in enumerate(sources):
                print(f"   Source {i+1}: {doc.metadata.get('source', 'Inconnue')}")
        
        # Analyser la qualité
        if answer and len(answer.strip()) > 20:
            if sources and len(sources) > 0:
                print("✅ La chaîne RAG fonctionne avec des sources")
                return True
            else:
                print("⚠️ La chaîne fonctionne mais sans sources (problème de retrieval)")
                return False
        else:
            print("❌ Réponse invalide de la chaîne RAG")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors du test de la chaîne RAG: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def analyze_logs():
    """Analyse des logs d'interaction"""
    print("\n🔍 Analyse des logs...")
    
    try:
        logs_dir = Path("logs")
        log_files = list(logs_dir.glob("interactions_*.jsonl"))
        
        if not log_files:
            print("❌ Aucun fichier de log trouvé")
            return False
        
        latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
        print(f"📂 Analyse du fichier: {latest_log.name}")
        
        interactions = []
        with open(latest_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    interaction = json.loads(line.strip())
                    interactions.append(interaction)
                except:
                    continue
        
        print(f"📊 Nombre d'interactions: {len(interactions)}")
        
        if interactions:
            # Analyser les dernières interactions
            for i, interaction in enumerate(interactions[-3:]):  # 3 dernières
                print(f"\n--- Interaction {len(interactions) - 2 + i} ---")
                print(f"❓ Question: {interaction.get('question', '')[:100]}...")
                print(f"🤖 Réponse: {interaction.get('answer', '')[:100]}...")
                sources = interaction.get('sources', [])
                print(f"📄 Sources: {len(sources)} -> {sources}")
                
                # Diagnostic
                if not sources:
                    print("⚠️ PROBLÈME: Aucune source utilisée")
                if len(interaction.get('answer', '')) < 50:
                    print("⚠️ PROBLÈME: Réponse très courte")
            
            return True
        else:
            print("❌ Aucune interaction valide trouvée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse des logs: {str(e)}")
        return False

def main():
    """Fonction principale de diagnostic"""
    print("🚀 === DIAGNOSTIC COMPLET DU CHATBOT RAG ===\n")
    
    results = {
        "vectordb": test_vectordb(),
        "retriever": test_retriever(), 
        "llm": test_llm(),
        "full_chain": test_full_chain(),
        "logs": analyze_logs()
    }
    
    print(f"\n📊 === RÉSUMÉ DU DIAGNOSTIC ===")
    for test_name, result in results.items():
        status = "✅ OK" if result else "❌ ÉCHEC"
        print(f"{test_name.upper()}: {status}")
    
    # Recommandations
    print(f"\n💡 === RECOMMANDATIONS ===")
    if not results["vectordb"]:
        print("🔧 Réindexer les documents dans la base vectorielle")
    if not results["retriever"]:
        print("🔧 Vérifier la configuration du retriever")
    if not results["llm"]:
        print("🔧 Vérifier le modèle LLM et sa configuration")
    if not results["full_chain"]:
        print("🔧 Revoir le prompt et la chaîne RAG")
    
    if all(results.values()):
        print("🎉 Tous les tests passent ! Le problème pourrait venir de l'évaluation elle-même.")
    else:
        print("🚨 Des problèmes ont été identifiés. Corrigez-les dans l'ordre indiqué.")

if __name__ == "__main__":
    main() 