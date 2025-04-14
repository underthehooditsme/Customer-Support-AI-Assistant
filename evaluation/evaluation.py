import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import json
from typing import List, Dict, Any, Tuple, Union, Optional
import logging
import time
from datetime import datetime
import os

try:
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = False # made false as need to implement
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS not installed. Some evaluation metrics will be unavailable.")


from models.embeddings import get_embeddings

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Class for evaluating RAG-based customer support responses using various metrics.
    """
    
    def __init__(self, embeddings_model=None, llm_model=None):
        """
        Initialize the evaluator with necessary models and metrics.
        
        Args:
            embeddings_model: Model to generate embeddings for semantic similarity
            llm_model: LLM model for LLM-based evaluation metrics
        """
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.embeddings_model = embeddings_model
        self.llm_model = llm_model
        
    
    def evaluate_response(self, 
                         query: str, 
                         generated_response: str, 
                         ground_truth: Union[str, List[str]] = None, 
                         retrieved_contexts: List[Dict[str, Any]] = None,
                         response_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensively evaluate a generated response against multiple criteria.
        
        Args:
            query: The original customer query
            generated_response: The RAG-generated response to evaluate
            ground_truth: Reference answer(s) to compare against (if available)
            retrieved_contexts: List of retrieved documents/contexts used for generation
            response_metadata: Additional metadata about the response generation process
            
        Returns:
            Dict containing all evaluation metrics
        """
        # Start timer for evaluation process
        start_time = time.time()
        
        evaluation_results = {
            "query": query,
            "generated_response": generated_response,
            "metrics": {},
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # Content-based metrics (when ground truth is available)
        if ground_truth:
            if isinstance(ground_truth, list):
                # Use the first ground truth for Rouge and semantic similarity
                primary_ground_truth = ground_truth[0]
                evaluation_results["metrics"].update(self._calculate_content_metrics(
                    generated_response, primary_ground_truth))
                
                # BLEU with multiple references
                evaluation_results["metrics"]["bleu_score"] = self._calculate_bleu(
                    generated_response, ground_truth)
            else:
                evaluation_results["metrics"].update(self._calculate_content_metrics(
                    generated_response, ground_truth))
                evaluation_results["metrics"]["bleu_score"] = self._calculate_bleu(
                    generated_response, [ground_truth])
        
        # Retrieval-based metrics
        if retrieved_contexts:
            evaluation_results["metrics"].update(self._calculate_retrieval_metrics(
                query, generated_response, retrieved_contexts))
        
        # Response quality metrics (independent of ground truth)
        evaluation_results["metrics"].update(self._calculate_quality_metrics(
            query, generated_response))
            
        # Response time and efficiency metrics
        if response_metadata and "response_time" in response_metadata:
            evaluation_results["metrics"]["response_time"] = response_metadata["response_time"]
        
        # RAGAS evaluation if available
        if retrieved_contexts:
            ragas_metrics = self._run_ragas_evaluation(
                query, 
                generated_response, 
                ground_truth,
                retrieved_contexts
            )
            if ragas_metrics:
                evaluation_results["metrics"].update(ragas_metrics)
        
        
        # Record evaluation time
        evaluation_results["evaluation_time"] = time.time() - start_time
        
        return evaluation_results
    
    def _calculate_content_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate content-based metrics comparing generated response to reference."""
        metrics = {}
        
        # ROUGE scores
        try:
            rouge_scores = self.rouge.get_scores(generated, reference)[0]
            metrics["rouge1_f"] = rouge_scores["rouge-1"]["f"]
            metrics["rouge2_f"] = rouge_scores["rouge-2"]["f"]
            metrics["rougeL_f"] = rouge_scores["rouge-l"]["f"]
        except Exception as e:
            logger.warning(f"Error calculating ROUGE scores: {e}")
            metrics["rouge1_f"] = metrics["rouge2_f"] = metrics["rougeL_f"] = 0.0
        
        # Semantic similarity using embeddings
        if self.embeddings_model:
            try:
                gen_embedding = get_embeddings(generated, self.embeddings_model)
                ref_embedding = get_embeddings(reference, self.embeddings_model)
                
                if len(gen_embedding.shape) == 1:
                    gen_embedding = gen_embedding.reshape(1, -1)
                if len(ref_embedding.shape) == 1:
                    ref_embedding = ref_embedding.reshape(1, -1)
                    
                similarity = cosine_similarity(gen_embedding, ref_embedding)[0][0]
                metrics["semantic_similarity"] = float(similarity)
            except Exception as e:
                logger.warning(f"Error calculating semantic similarity: {e}")
                metrics["semantic_similarity"] = 0.0
        
        return metrics
    
    def _calculate_bleu(self, generated: str, references: List[str]) -> float:
        """Calculate BLEU score for generated text against multiple references."""
        try:
            generated_tokens = generated.lower().split()
            reference_tokens = [ref.lower().split() for ref in references]
            return sentence_bleu(reference_tokens, generated_tokens, smoothing_function=self.smooth)
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def _calculate_retrieval_metrics(self, query: str, generated: str, 
                                   contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics related to retrieved contexts and their usage."""
        metrics = {}
        
        # Context relevance score (proxy through query-context similarity)
        if self.embeddings_model and contexts:
            try:
                query_embedding = get_embeddings(query, self.embeddings_model)
                context_embeddings = [get_embeddings(ctx.get("text", ""), self.embeddings_model) 
                                    for ctx in contexts if "text" in ctx]
                
                if context_embeddings:
                    # Reshape embeddings if necessary
                    if len(query_embedding.shape) == 1:
                        query_embedding = query_embedding.reshape(1, -1)
                    
                    # Calculate similarities
                    similarities = [
                        float(cosine_similarity(
                            query_embedding, 
                            emb.reshape(1, -1) if len(emb.shape) == 1 else emb
                        )[0][0])
                        for emb in context_embeddings
                    ]
                    
                    metrics["avg_context_relevance"] = np.mean(similarities)
                    metrics["max_context_relevance"] = np.max(similarities)
            except Exception as e:
                logger.warning(f"Error calculating context relevance: {e}")
        
        # Response-context overlap (how much of the response seems derived from contexts)
        try:
            context_texts = " ".join([ctx.get("text", "") for ctx in contexts if "text" in ctx])
            
            # Simple word overlap ratio
            context_words = set(context_texts.lower().split())
            response_words = set(generated.lower().split())
            if response_words:
                metrics["context_utilization"] = len(context_words.intersection(response_words)) / len(response_words)
            else:
                metrics["context_utilization"] = 0.0
        except Exception as e:
            logger.warning(f"Error calculating context utilization: {e}")
            metrics["context_utilization"] = 0.0
            
        return metrics
    
    def _calculate_quality_metrics(self, query: str, response: str) -> Dict[str, Any]:
        """Calculate metrics for intrinsic response quality."""
        metrics = {}
        
        # Response length (proxy for informativeness)
        metrics["response_length"] = len(response.split())
        
        # Readability metrics (simple proxy via average sentence length)
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            metrics["avg_sentence_length"] = float(avg_sentence_length)
        
        # Query terms coverage (how many query terms appear in response)
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        if query_terms:
            metrics["query_term_coverage"] = len(query_terms.intersection(response_terms)) / len(query_terms)
        else:
            metrics["query_term_coverage"] = 0.0
            
        return metrics
    
    def _run_ragas_evaluation(self, query: str, response: str, 
                            contexts: List[Dict[str, Any]],ground_truth: str) -> Optional[Dict[str, float]]:
        """
        Run RAGAS evaluation metrics if available.
        
        Args:
            query: The customer query
            response: The generated response
            contexts: Retrieved contexts used for generation
            ground_truth: Refrence data
            
        Returns:
            Dictionary of RAGAS metrics or None if unavailable
        """
        if not RAGAS_AVAILABLE:
            return None


    
    def batch_evaluate(self, evaluation_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Evaluate multiple responses and return results as a DataFrame.
        
        Args:
            evaluation_data: List of dictionaries containing query, response, ground_truth, etc.
            
        Returns:
            DataFrame containing all evaluation results
        """
        results = []
        
        for item in evaluation_data:
            result = self.evaluate_response(
                query=item.get("query", ""),
                generated_response=item.get("generated_response", ""),
                ground_truth=item.get("ground_truth"),
                retrieved_contexts=item.get("retrieved_contexts"),
                response_metadata=item.get("response_metadata")
            )
            
            # Flatten metrics for DataFrame
            flat_result = {
                "query": result["query"],
                "generated_response": result["generated_response"]
            }
            
            if "ground_truth" in item:
                flat_result["ground_truth"] = item["ground_truth"] if isinstance(item["ground_truth"], str) else item["ground_truth"][0]
                
            # Add all metrics with their prefix
            for metric_name, metric_value in result["metrics"].items():
                flat_result[f"metric_{metric_name}"] = metric_value
                
            results.append(flat_result)
            
        return pd.DataFrame(results)
    
    def calculate_overall_metrics(self, evaluation_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate aggregated metrics across all evaluated responses.
        
        Args:
            evaluation_df: DataFrame with evaluation results from batch_evaluate
            
        Returns:
            Dictionary of averaged metrics
        """
        metric_columns = [col for col in evaluation_df.columns if col.startswith("metric_")]
        
        if not metric_columns:
            return {}
            
        overall_metrics = {}
        for col in metric_columns:
            # Remove 'metric_' prefix for cleaner keys
            clean_name = col[7:]  # Remove 'metric_' prefix
            overall_metrics[f"avg_{clean_name}"] = float(evaluation_df[col].mean())
            
        return overall_metrics
    
    def save_evaluation_results(self, results_df: pd.DataFrame, file_path: str, 
                              include_overall: bool = True) -> None:
        """
        Save evaluation results to file (CSV, JSON).
        
        Args:
            results_df: DataFrame with evaluation results
            file_path: Path to save results
            include_overall: Whether to include overall metrics
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if file_path.endswith('.csv'):
            results_df.to_csv(file_path, index=False)
        elif file_path.endswith('.json'):
            results_dict = results_df.to_dict(orient='records')
            
            if include_overall:
                overall_metrics = self.calculate_overall_metrics(results_df)
                output = {
                    "individual_results": results_dict,
                    "overall_metrics": overall_metrics
                }
            else:
                output = results_dict
                
            with open(file_path, 'w') as f:
                json.dump(output, f, indent=2)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
def create_test_data():
    """Create sample data for testing the RAG evaluator"""
    
    # Sample customer support queries
    test_data = [
        {
            "query": "How do I reset my password?",
            "generated_response": "To reset your password, go to the login page and click on 'Forgot Password'. You'll receive an email with instructions to create a new password.",
            "ground_truth": "You can reset your password by clicking the 'Forgot Password' link on the login page. Follow the email instructions to create a new password.",
            "retrieved_contexts": [
                {
                    "text": "Password reset procedure: Users who forget their password can click on the 'Forgot Password' link on the login screen. The system will send an email to the registered email address with a link to reset the password.",
                    "score": 0.92
                },
                {
                    "text": "User authentication in our system uses secure password hashing and storage. All passwords are encrypted and not stored in plain text.",
                    "score": 0.65
                }
            ],
            "response_metadata": {
                "response_time": 0.8
            }
        },
        {
            "query": "Can I cancel my subscription?",
            "generated_response": "Yes, you can cancel your subscription at any time. Go to Account Settings, select Subscriptions, and click on Cancel Subscription. Your access will continue until the end of the billing period.",
            "ground_truth": "Subscription cancellation is available through your Account Settings page. Navigate to the Subscriptions tab and select Cancel. You'll retain access until the current billing cycle ends.",
            "retrieved_contexts": [
                {
                    "text": "Subscription cancellation policy: Users can cancel their subscription at any time through their Account Settings. After cancellation, they will maintain access to premium features until the end of their current billing cycle.",
                    "score": 0.89
                },
                {
                    "text": "To cancel a subscription: 1. Log in to your account. 2. Go to Account Settings. 3. Select the Subscriptions tab. 4. Click on Cancel Subscription button. 5. Confirm cancellation.",
                    "score": 0.93
                }
            ],
            "response_metadata": {
                "response_time": 0.6
            }
        },
        {
            "query": "What payment methods do you accept?",
            "generated_response": "We accept credit cards (Visa, MasterCard, American Express), PayPal, and bank transfers. For business accounts, we also offer invoice payment options.",
            "ground_truth": "Our platform accepts all major credit cards including Visa, MasterCard, and American Express. We also support PayPal and direct bank transfers. Enterprise customers can request invoice-based billing.",
            "retrieved_contexts": [
                {
                    "text": "Payment methods: Our system currently supports payments via credit cards (Visa, MasterCard, American Express, Discover), PayPal, and ACH bank transfers.",
                    "score": 0.95
                },
                {
                    "text": "For enterprise customers, we offer additional payment options including invoicing with net-30 payment terms. Please contact your account representative to set up this payment method.",
                    "score": 0.72
                }
            ],
            "response_metadata": {
                "response_time": 0.5
            }
        }
    ]
    
    return test_data


def main():
    """Main function to demonstrate RAG evaluation"""
    
    logger.info("Starting RAG evaluation demonstration")
    
    # Initialize evaluator
    # In a real scenario, you would provide actual embedding and LLM models
    evaluator = RAGEvaluator()
    
    # Generate test data
    test_data = create_test_data()
    logger.info(f"Created {len(test_data)} test examples")
    
    # Single evaluation demo
    single_example = test_data[0]
    logger.info(f"Running single evaluation for query: '{single_example['query']}'")
    
    single_result = evaluator.evaluate_response(
        query=single_example["query"],
        generated_response=single_example["generated_response"],
        ground_truth=single_example["ground_truth"],
        retrieved_contexts=single_example["retrieved_contexts"],
        response_metadata=single_example["response_metadata"]
    )
    
    print("\n--- Single Evaluation Results ---")
    print(f"Query: {single_result['query']}")
    print(f"Response: {single_result['generated_response']}")
    print("\nMetrics:")
    for metric, value in single_result["metrics"].items():
        print(f"{metric}: {value:.4f}")
    
    # Batch evaluation demo
    logger.info("Running batch evaluation")
    results_df = evaluator.batch_evaluate(test_data)
    
    print("\n--- Batch Evaluation Summary ---")
    overall_metrics = evaluator.calculate_overall_metrics(results_df)
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results demo
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    json_path = os.path.join(output_dir, f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    logger.info(f"Saving results to CSV: {csv_path}")
    evaluator.save_evaluation_results(results_df, csv_path)
    
    logger.info(f"Saving results to JSON: {json_path}")
    evaluator.save_evaluation_results(results_df, json_path)
    
    print(f"\nResults saved to {csv_path} and {json_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
