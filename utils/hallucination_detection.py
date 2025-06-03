"""
Module de détection et correction avancée des hallucinations.
Implémente des méthodes sophistiquées pour assurer la fiabilité des réponses.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Résultat de la validation d'une réponse."""
    is_valid: bool
    confidence_score: float
    feedback: Dict[str, float]
    suggested_correction: Optional[str] = None

class HallucinationDetector:
    def __init__(self):
        """Initialise le détecteur d'hallucinations avec le modèle de similarité."""
        try:
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Modèle de similarité chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            self.encoder = None

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité sémantique entre deux textes."""
        try:
            if not self.encoder:
                return 0.0
            embeddings = self.encoder.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Erreur lors du calcul de similarité: {str(e)}")
            return 0.0

    def get_dynamic_threshold(self, question_type: str, question_length: int) -> float:
        """Détermine le seuil de confiance dynamique basé sur le type de question."""
        base_threshold = 0.7
        
        # Ajustement selon le type de question
        type_adjustments = {
            'factuel': 0.8,    # Questions factuelles nécessitent plus de précision
            'opinion': 0.6,    # Questions d'opinion peuvent être plus flexibles
            'technique': 0.75,  # Questions techniques nécessitent une bonne précision
            'général': 0.65    # Questions générales sont plus flexibles
        }
        
        # Ajustement selon la longueur de la question
        length_factor = min(1.0, question_length / 100)  # Questions plus longues = seuil plus bas
        
        threshold = base_threshold * type_adjustments.get(question_type, 1.0) * (1 - length_factor * 0.2)
        return min(0.9, max(0.5, threshold))  # Garder entre 0.5 et 0.9

    def check_logical_consistency(self, answer: str, context: str) -> Tuple[bool, float]:
        """Vérifie la cohérence logique de la réponse par rapport au contexte."""
        try:
            # Découper en phrases
            answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
            context_sentences = [s.strip() for s in context.split('.') if s.strip()]
            
            # Vérifier la cohérence entre phrases
            contradictions = 0
            total_comparisons = 0
            
            for i, sent1 in enumerate(answer_sentences):
                # Cohérence interne
                if i > 0:
                    similarity = self.calculate_semantic_similarity(answer_sentences[i-1], sent1)
                    if similarity < 0.2:  # Très faible cohérence entre phrases consécutives
                        contradictions += 1
                    total_comparisons += 1
                
                # Cohérence avec le contexte
                context_similarities = [
                    self.calculate_semantic_similarity(sent1, ctx_sent)
                    for ctx_sent in context_sentences
                ]
                if max(context_similarities, default=0) < 0.3:  # Aucune phrase du contexte ne supporte cette affirmation
                    contradictions += 1
                total_comparisons += 1
            
            consistency_score = 1.0 - (contradictions / total_comparisons if total_comparisons > 0 else 0)
            return consistency_score > 0.7, consistency_score
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de cohérence: {str(e)}")
            return False, 0.0

    def validate_response(self, 
                         answer: str, 
                         context: str, 
                         question: str,
                         question_type: str = 'général') -> ValidationResult:
        """
        Valide une réponse avec plusieurs critères et retourne un score de confiance.
        
        Args:
            answer: La réponse à valider
            context: Le contexte utilisé pour générer la réponse
            question: La question posée
            question_type: Le type de question (factuel, opinion, technique, général)
            
        Returns:
            ValidationResult contenant le résultat de la validation
        """
        try:
            # 1. Calcul des scores individuels
            semantic_score = self.calculate_semantic_similarity(answer, context)
            question_relevance = self.calculate_semantic_similarity(question, answer)
            is_consistent, consistency_score = self.check_logical_consistency(answer, context)
            
            # 2. Calcul du seuil dynamique
            threshold = self.get_dynamic_threshold(question_type, len(question))
            
            # 3. Pondération des scores
            weights = {
                'semantic': 0.4,
                'relevance': 0.3,
                'consistency': 0.3
            }
            
            final_score = (
                semantic_score * weights['semantic'] +
                question_relevance * weights['relevance'] +
                consistency_score * weights['consistency']
            )
            
            # 4. Préparation du feedback
            feedback = {
                'semantic_similarity': semantic_score,
                'question_relevance': question_relevance,
                'logical_consistency': consistency_score,
                'threshold_used': threshold
            }
            
            # 5. Décision et correction si nécessaire
            is_valid = final_score >= threshold
            suggested_correction = None
            
            if not is_valid:
                # Identifier la partie la plus fiable de la réponse
                answer_sentences = [s.strip() for s in answer.split('.') if s.strip()]
                best_sentence = max(answer_sentences, 
                                  key=lambda s: self.calculate_semantic_similarity(s, context))
                suggested_correction = best_sentence + "."
            
            return ValidationResult(
                is_valid=is_valid,
                confidence_score=final_score,
                feedback=feedback,
                suggested_correction=suggested_correction
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                confidence_score=0.0,
                feedback={'error': str(e)},
                suggested_correction=None
            )

    def log_validation_result(self, result: ValidationResult, metadata: Dict) -> None:
        """Enregistre les résultats de validation pour analyse ultérieure."""
        try:
            log_entry = {
                'timestamp': metadata.get('timestamp'),
                'validation_result': {
                    'is_valid': result.is_valid,
                    'confidence_score': result.confidence_score,
                    'feedback': result.feedback
                },
                'metadata': metadata
            }
            
            logger.info(f"Résultat de validation: {log_entry}")
            
            # TODO: Implémenter la persistance des logs pour analyse
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du résultat: {str(e)}") 