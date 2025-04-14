import logging
from typing import Dict, Any, List, Optional
import numpy as np
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class ExplainabilityTool:
    """Tool for providing explainability information for RAG responses."""
    
    def __init__(self):
        """Initialize the explainability tool."""
        logger.info("Initializing explainability tool")
    
    def explain(
        self, 
        query: str, 
        response: str, 
        context_docs: List[Document], 
        similarity_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Generate explainability data for a RAG response.
        
        Args:
            query: The user's query
            response: The generated response
            context_docs: The retrieved documents used as context
            similarity_scores: Similarity scores for the retrieved documents
            
        Returns:
            Dictionary with explainability information
        """
        logger.info(f"Generating explainability data for query: {query}")
        
        try:
            # Calculate overall confidence score
            confidence_score = self._calculate_confidence_score(similarity_scores, response)
            
            # Identify key information from context documents
            key_information = self._identify_key_information(context_docs, response)
            
            # Format context documents
            formatted_docs = self._format_context_documents(context_docs, similarity_scores)
            
            # Create a reasoning trace
            reasoning_trace = self._create_reasoning_trace(query, context_docs, response)
            
            # Calculate a faithfulness score (how much the response relies on context)
            faithfulness_score = self._calculate_faithfulness(response, context_docs)
            
            # Prepare explainability data
            explainability_data = {
                "confidence_score": confidence_score,
                "key_information": key_information,
                "context_documents": formatted_docs,
                "reasoning_trace": reasoning_trace,
                "faithfulness_score": faithfulness_score
            }
            
            logger.info("Explainability data generated successfully")
            return explainability_data
            
        except Exception as e:
            logger.error(f"Error generating explainability data: {e}")
            return {
                "error": f"Error generating explainability data: {str(e)}"
            }
    
    def _calculate_confidence_score(self, similarity_scores: List[float], response: str) -> float:
        """
        Calculate a confidence score for the RAG response.
        
        Args:
            similarity_scores: Similarity scores for the retrieved documents
            response: The generated response
            
        Returns:
            Confidence score (0-100)
        """
        # Base confidence from similarity scores
        if not similarity_scores:
            return 0.0  
        
        similarity_scores_normalized = [1.0 - score for score in similarity_scores]
        
        # Calculate weighted average based on position (first doc more important)
        weights = [1.0, 0.7, 0.5, 0.3, 0.2] + [0.1] * (len(similarity_scores) - 5)
        weights = weights[:len(similarity_scores)]
        weighted_sum = sum(w * s for w, s in zip(weights, similarity_scores_normalized))
        base_confidence = (weighted_sum / sum(weights)) * 70  # 70% of score from similarity
        

        response_length = min(len(response), 300) / 300.0 * 30
        

        confidence = base_confidence + response_length
        
        return min(100.0, max(0.0, confidence))  # Clamp to 0-100 range
    
    def _identify_key_information(self, context_docs: List[Document], response: str) -> List[Dict[str, Any]]:
        """
        Identify key pieces of information from context documents used in the response.
        
        Args:
            context_docs: The retrieved documents used as context
            response: The generated response
            
        Returns:
            List of key information pieces
        """
        key_info = []
        
        if not context_docs:
            return key_info
        
        response_sentences = response.split(". ")
        
        for i, doc in enumerate(context_docs):
            # Split document into sentences
            doc_sentences = doc.page_content.split(". ")
            
            # Find potential matching information
            for doc_sentence in doc_sentences:
                if len(doc_sentence) < 10:  
                    continue
                
                # Check if this sentence or similar information appears in the response
                for response_sentence in response_sentences:
                    # Simple similarity measure (could be improved with embeddings)
                    words_doc = set(doc_sentence.lower().split())
                    words_response = set(response_sentence.lower().split())
                    
                    if len(words_doc) > 0 and len(words_response) > 0:
                        overlap = len(words_doc.intersection(words_response)) / min(len(words_doc), len(words_response))
                        
                        if overlap > 0.3:  # Threshold for considering a match
                            key_info.append({
                                "doc_index": i,
                                "text": doc_sentence,
                                "overlap_score": round(overlap * 100, 1)
                            })
                            break
        
        return sorted(key_info, key=lambda x: x["overlap_score"], reverse=True)[:3]
    
    def _format_context_documents(
        self, 
        context_docs: List[Document], 
        similarity_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Format context documents for the explainability data.
        
        Args:
            context_docs: The retrieved documents used as context
            similarity_scores: Similarity scores for the retrieved documents
            
        Returns:
            List of formatted documents
        """
        formatted_docs = []
        
        for i, (doc, score) in enumerate(zip(context_docs, similarity_scores)):
            metadata = {}
            if hasattr(doc, "metadata"):
                for key, value in doc.metadata.items():
                    # Limiting string length in metadata(for future have added)
                    if isinstance(value, str) and len(value) > 100:
                        metadata[key] = value[:100] + "..."
                    else:
                        metadata[key] = value
            

            formatted_docs.append({
                "index": i + 1,
                "relevance_score": round((1.0 - score) * 100, 1),  # Convert distance to relevance
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": metadata
            })
        
        return formatted_docs
    
    def _create_reasoning_trace(
        self, 
        query: str, 
        context_docs: List[Document], 
        response: str
    ) -> List[str]:
        """
        Create a reasoning trace explaining how the response was generated.
        
        Args:
            query: The user's query
            context_docs: The retrieved documents used as context
            response: The generated response
            
        Returns:
            List of reasoning steps
        """
        # A simple reasoning trace
        reasoning_trace = [
            f"1. Received query: \"{query}\"",
            f"2. Retrieved {len(context_docs)} relevant documents from the knowledge base"
        ]
        
        for i, doc in enumerate(context_docs[:3]): 
            # A brief summary
            summary = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            reasoning_trace.append(f"3.{i+1}. Document {i+1} contains information: \"{summary}\"")
        
        # Add information about response generation
        reasoning_trace.append(f"4. Generated response based on the retrieved information")
        
        return reasoning_trace
    
    def _calculate_faithfulness(self, response: str, context_docs: List[Document]) -> float:
        """
        Calculate a faithfulness score for the response based on context.
        
        Args:
            response: The generated response
            context_docs: The retrieved documents used as context
            
        Returns:
            Faithfulness score (0-100)
        """
        if not context_docs:
            return 50.0  
        
        response_words = set(response.lower().split())
        

        context_text = " ".join([doc.page_content for doc in context_docs])
        context_words = set(context_text.lower().split())
        

        if len(response_words) == 0:
            return 0.0
        
        overlap_ratio = len(response_words.intersection(context_words)) / len(response_words)
        
        faithfulness_score = overlap_ratio * 100
        
        return min(100.0, max(0.0, faithfulness_score))  

    def highlight_context_in_response(self, response: str, context_docs: List[Document]) -> str:
        """
        Highlight parts of the response that are derived from context.
        
        Args:
            response: The generated response
            context_docs: The retrieved documents used as context
            
        Returns:
            Response with HTML highlighting of context-derived parts
        """
        logger.info("Highlighting context in response")
        
        if not context_docs or not response:
            return response
        
        # This is a simplified implementation - a more sophisticated approach would use
        # semantic similarity and better text chunking
        
        # Combine context documents
        context_text = " ".join([doc.page_content for doc in context_docs])
        
        # Split response into sentences
        response_sentences = response.split(". ")
        highlighted_sentences = []
        
        for sentence in response_sentences:
            # Simple word-based similarity check
            sentence_words = set(sentence.lower().split())
            context_words = set(context_text.lower().split())
            
            if len(sentence_words) > 0:
                overlap = len(sentence_words.intersection(context_words)) / len(sentence_words)
                
                if overlap > 0.3:  # Threshold for considering context-derived
                    highlighted_sentences.append(f"<mark>{sentence}</mark>")
                else:
                    highlighted_sentences.append(sentence)
        
        # Join sentences back together
        highlighted_response = ". ".join(highlighted_sentences)
        
        return highlighted_response


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test 
    tool = ExplainabilityTool()
    
    # Mock data
    query = "My laptop arrived with a broken screen. What should I do?"
    response = "I'm sorry to hear about your laptop with a broken screen. You should contact our customer support team within 14 days of delivery to report the damage. You can do this by email or phone. Please have your order number ready. We'll likely arrange for a return and replacement as this falls under warranty for damaged-on-arrival items."
    
    context_docs = [
        Document(
            page_content="For damaged items, customers should report to customer support within 14 days of delivery. This includes broken screens, malfunctioning keyboards, or other hardware issues. Contact via phone or email with your order number ready.",
            metadata={"query": "What do I do if my new laptop is damaged?", "response_id": "12345"}
        ),
        Document(
            page_content="Our warranty covers manufacturer defects and items damaged during shipping. This includes broken screens if reported within the return period. We typically process replacements within 5-7 business days after receiving the returned item.",
            metadata={"query": "Is a broken screen covered by warranty?", "response_id": "23456"}
        ),
        Document(
            page_content="To return a damaged laptop, first contact customer support to report the issue. They will provide a return label and instructions. Pack the item securely in its original packaging if possible. Return shipping for damaged-on-arrival items is free.",
            metadata={"query": "How do I return a damaged laptop?", "response_id": "34567"}
        )
    ]
    
    similarity_scores = [0.15, 0.22, 0.31]  # Lower is better (distances)
    
    # Get explainability data
    explainability_data = tool.explain(query, response, context_docs, similarity_scores)
    
    # Print the explainability data
    import json
    print(json.dumps(explainability_data, indent=2))
    
    # Highlighting
    highlighted_response = tool.highlight_context_in_response(response, context_docs)
    print("\nHighlighted Response:")
    print(highlighted_response)
