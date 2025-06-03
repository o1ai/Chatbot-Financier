"""
Évaluation RAG simplifiée - Version compatible
Système d'évaluation sans conflit de métaclasse
"""

import os
import json
from datetime import datetime as dt
from pathlib import Path
import streamlit as st
from typing import List, Dict, Any
import numpy as np

# Configuration
LOGS_DIR = Path("logs")
EVALUATION_DIR = Path("evaluations")
LOGS_DIR.mkdir(exist_ok=True)
EVALUATION_DIR.mkdir(exist_ok=True)

class SimpleRAGEvaluator:
    """Évaluateur RAG simplifié sans conflits de métaclasse"""
    
    def __init__(self):
        self.api_key = None
    
    def get_latest_log_file(self):
        """Récupère le fichier de log le plus récent"""
        log_files = list(LOGS_DIR.glob("interactions_*.jsonl"))
        if not log_files:
            st.warning("⚠️ Aucun fichier de log trouvé")
            return None
        return max(log_files, key=os.path.getmtime)
    
    def prepare_dataset(self, log_file, max_samples=50):
        """Prépare le dataset pour l'évaluation"""
        samples = []
        
        if not log_file or not log_file.exists():
            st.error("❌ Fichier de log introuvable")
            return None
        
        with open(log_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    interaction = json.loads(line.strip())
                    
                    sample = {
                        "question": interaction.get("question", ""),
                        "answer": interaction.get("answer", ""),
                        "contexts": interaction.get("sources", []),
                        "ground_truth": interaction.get("reference", "")
                    }
                    
                    if sample["question"] and sample["answer"]:
                        samples.append(sample)
                        
                except json.JSONDecodeError:
                    continue
        
        if not samples:
            st.error("❌ Aucune interaction valide trouvée")
            return None
        
        st.info(f"📊 {len(samples)} interactions préparées pour l'évaluation")
        return samples
    
    def calculate_simple_metrics(self, samples: List[Dict]) -> Dict[str, float]:
        """Calcule des métriques simples équivalentes à RAGAS"""
        
        faithfulness_scores = []
        relevancy_scores = []
        precision_scores = []
        recall_scores = []
        
        for sample in samples:
            question = sample["question"].lower()
            answer = sample["answer"].lower()
            contexts = [ctx.lower() for ctx in sample["contexts"]]
            
            # 1. Faithfulness (fidélité aux sources)
            if contexts:
                # Calculer combien de mots de la réponse viennent des contextes
                answer_words = set(answer.split())
                context_words = set()
                for ctx in contexts:
                    context_words.update(ctx.split())
                
                if answer_words:
                    faithful_words = len(answer_words & context_words)
                    faithfulness = faithful_words / len(answer_words)
                else:
                    faithfulness = 0.0
            else:
                faithfulness = 0.0
            
            faithfulness_scores.append(min(faithfulness, 1.0))
            
            # 2. Answer Relevancy (pertinence de la réponse)
            question_words = set(question.split())
            answer_words = set(answer.split())
            
            if question_words and answer_words:
                common_words = len(question_words & answer_words)
                relevancy = common_words / min(len(question_words), len(answer_words))
            else:
                relevancy = 0.0
            
            relevancy_scores.append(min(relevancy, 1.0))
            
            # 3. Context Precision (précision du contexte)
            if contexts and question_words:
                relevant_contexts = 0
                for ctx in contexts:
                    ctx_words = set(ctx.split())
                    if question_words & ctx_words:
                        relevant_contexts += 1
                
                precision = relevant_contexts / len(contexts) if contexts else 0.0
            else:
                precision = 0.0
            
            precision_scores.append(precision)
            
            # 4. Context Recall (rappel du contexte)
            if contexts and answer_words:
                context_coverage = 0
                total_context_words = set()
                for ctx in contexts:
                    total_context_words.update(ctx.split())
                
                if total_context_words:
                    covered_words = len(answer_words & total_context_words)
                    recall = covered_words / len(total_context_words)
                else:
                    recall = 0.0
            else:
                recall = 0.0
            
            recall_scores.append(min(recall, 1.0))
        
        # Moyennes
        return {
            "faithfulness": np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            "answer_relevancy": np.mean(relevancy_scores) if relevancy_scores else 0.0,
            "context_precision": np.mean(precision_scores) if precision_scores else 0.0,
            "context_recall": np.mean(recall_scores) if recall_scores else 0.0
        }
    
    def run_evaluation(self, samples):
        """Exécute l'évaluation simplifiée"""
        try:
            st.info("🔄 Évaluation en cours...")
            progress_bar = st.progress(0)
            
            # Calcul des métriques
            results = self.calculate_simple_metrics(samples)
            
            progress_bar.progress(100)
            st.success("✅ Évaluation terminée !")
            
            return results
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'évaluation: {str(e)}")
            return None
    
    def save_results(self, results, dataset_size):
        """Sauvegarde les résultats"""
        timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = EVALUATION_DIR / f"simple_eval_{timestamp}.json"
        
        try:
            evaluation_data = {
                "timestamp": timestamp,
                "dataset_size": dataset_size,
                "metrics": results,
                "evaluation_type": "simple_rag"
            }
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"💾 Résultats sauvegardés: {output_file}")
            return output_file
            
        except Exception as e:
            st.error(f"❌ Erreur de sauvegarde: {str(e)}")
            return None
    
    def display_results(self, results):
        """Affiche les résultats de manière claire"""
        if not results:
            return
        
        st.subheader("📊 Résultats de l'évaluation RAG")
        
        try:
            # Affichage en colonnes
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="🎯 Faithfulness", 
                    value=f"{results.get('faithfulness', 0):.3f}",
                    help="Fidélité aux sources (0-1)"
                )
                st.metric(
                    label="📝 Answer Relevancy", 
                    value=f"{results.get('answer_relevancy', 0):.3f}",
                    help="Pertinence des réponses (0-1)"
                )
            
            with col2:
                st.metric(
                    label="🎯 Context Precision", 
                    value=f"{results.get('context_precision', 0):.3f}",
                    help="Précision du contexte (0-1)"
                )
                st.metric(
                    label="🔍 Context Recall", 
                    value=f"{results.get('context_recall', 0):.3f}",
                    help="Rappel du contexte (0-1)"
                )
            
            # Score global
            global_score = sum(results.values()) / len(results)
            st.metric(
                label="🏆 Score Global", 
                value=f"{global_score:.3f}",
                help="Moyenne de toutes les métriques"
            )
            
            # Interprétation
            if global_score >= 0.8:
                st.success("🌟 Excellent ! Votre chatbot performe très bien.")
            elif global_score >= 0.6:
                st.info("👍 Bon ! Il y a quelques points d'amélioration.")
            elif global_score >= 0.4:
                st.warning("⚠️ Moyen. Des améliorations sont nécessaires.")
            else:
                st.error("🔴 Faible. Révision majeure recommandée.")
            
            # Détails des métriques
            with st.expander("📋 Détails des métriques"):
                st.markdown("""
                **Faithfulness (Fidélité):** Mesure si les réponses sont basées sur les sources fournies.
                - Score élevé = peu d'hallucinations
                - Score faible = réponses inventées
                
                **Answer Relevancy (Pertinence):** Mesure si les réponses correspondent aux questions.
                - Score élevé = réponses pertinentes
                - Score faible = réponses hors-sujet
                
                **Context Precision (Précision):** Mesure la qualité du contexte récupéré.
                - Score élevé = contexte pertinent
                - Score faible = contexte non pertinent
                
                **Context Recall (Rappel):** Mesure si tout le contexte pertinent est utilisé.
                - Score élevé = utilisation complète du contexte
                - Score faible = contexte sous-utilisé
                """)
            
        except Exception as e:
            st.error(f"❌ Erreur d'affichage: {str(e)}")

def main():
    """Interface principale Streamlit"""
    st.title("🎯 Évaluation RAG Simplifiée")
    st.markdown("**Évaluation compatible sans conflits techniques**")
    
    # Information
    st.info("""
    🔬 **Évaluation RAG Simplifiée**
    
    Métriques équivalentes à RAGAS mais calculées de manière algorithmique :
    - **🎯 Faithfulness** : Fidélité aux sources (algorithme de recouvrement de mots)
    - **📝 Answer Relevancy** : Pertinence (similarité question-réponse)  
    - **🎯 Context Precision** : Qualité du contexte (pertinence des sources)
    - **🔍 Context Recall** : Couverture du contexte (utilisation complète)
    
    ✅ **Avantages :** Rapide, fiable, sans dépendances complexes
    """)
    
    # Configuration
    max_samples = st.number_input(
        "📊 Nombre max d'interactions",
        min_value=5,
        max_value=500,
        value=100,
        help="Nombre d'interactions à analyser"
    )
    
    # Bouton d'évaluation
    if st.button("🚀 Lancer l'évaluation RAG", type="primary", use_container_width=True):
        
        evaluator = SimpleRAGEvaluator()
        
        # Récupération des logs
        log_file = evaluator.get_latest_log_file()
        if not log_file:
            st.error("❌ Aucun fichier de log trouvé. Utilisez d'abord le chatbot.")
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
        
        st.balloons()

if __name__ == "__main__":
    main() 