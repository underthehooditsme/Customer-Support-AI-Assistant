"""RAG pipeline using LangGraph for customer support."""
import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_groq import ChatGroq
import langgraph.graph as lg
from langgraph.graph import END, StateGraph
from langchain_community.callbacks import get_openai_callback

from models.embeddings import EmbeddingManager
from models.llm_manager import LLMManager
from utils.explainability import ExplainabilityTool
from database.db_manager import DatabaseManager

from config import CONFIG

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# State for graph
class RAGState(BaseModel):
    """State for the RAG pipeline."""
    query: str = Field(description="The user's query")
    context: List[Document] = Field(default_factory=list, description="Retrieved documents")
    response: Optional[str] = Field(default=None, description="Generated response")
    scores: List[float] = Field(default_factory=list, description="Similarity scores")
    explainability: Dict[str, Any] = Field(default_factory=dict, description="Explainability data")
    error: Optional[str] = Field(default=None, description="Error message if any")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID")
    feedback: Optional[Dict[str, Any]] = Field(default=None, description="User feedback")

class RAGPipeline:
    """RAG pipeline for customer support using LangGraph."""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        llm_manager: LLMManager,
        explainability_tool: Optional[ExplainabilityTool] = None,
        db_manager: Optional[DatabaseManager] = None,
        top_k: int = 3
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_manager: Manager for embeddings and vector store
            llm_manager: Manager for LLM models
            explainability_tool: Tool for explainability
            db_manager: Database manager for storing interactions
            top_k: Number of documents to retrieve
        """
        logger.info("Initializing RAG pipeline")
        self.embedding_manager = embedding_manager
        self.llm_manager = llm_manager
        self.explainability_tool = explainability_tool
        self.db_manager = db_manager
        self.top_k = top_k
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph for the RAG pipeline.
        
        Returns:
            The constructed StateGraph
        """

        builder = StateGraph(RAGState)
        
        builder.add_node("retrieve", self._retrieve_documents)
        builder.add_node("generate", self._generate_response)
        builder.add_node("explain", self._add_explainability)
        builder.add_node("store", self._store_interaction)
        
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", "explain")
        builder.add_edge("explain", "store")
        builder.add_edge("store", END)
        
        builder.set_entry_point("retrieve")
        
        graph = builder.compile()
        logger.info("RAG pipeline graph built successfully")
        
        return graph
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with retrieved documents
        """
        logger.info(f"Retrieving documents for query: {state.query}")
        
        try:
            docs_with_scores = self.embedding_manager.get_similar_documents_with_scores(
                state.query, k=self.top_k
            )
            
            docs = [doc for doc, _ in docs_with_scores]
            scores = [float(score) for _, score in docs_with_scores]
            
            state.context = docs
            state.scores = scores
            
            logger.info(f"Retrieved {len(docs)} documents")
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            state.error = f"Error retrieving documents: {str(e)}"
        
        return state
    
    def _generate_response(self, state: RAGState) -> RAGState:
        """
        Generate a response using the LLM based on retrieved documents.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with generated response
        """
        logger.info("Generating response based on retrieved documents")
        
        try:
            # Check if we have an error from previous steps
            if state.error:
                return state
            
            # Generate system prompt based on context
            system_prompt = """You are a helpful customer support assistant. Using the provided context from previous 
            support conversations, generate a helpful, accurate, and professional response to the user's query.
            If the context doesn't provide enough information to answer the question properly, provide a helpful
            general response but be honest about what you don't know. 
            
            Remember to:
            1. Be empathetic to the customer's problem
            2. Provide clear steps or solutions when possible
            3. Maintain a professional, friendly tone
            4. Only use information from the provided context
            """
            
            # Generate response
            result = self.llm_manager.generate_response(
                query=state.query,
                retrieved_docs=state.context,
                system_prompt=system_prompt
            )

            state.response = result["response"]
            
            logger.info("Response generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            state.error = f"Error generating response: {str(e)}"
        
        return state
    
    def _add_explainability(self, state: RAGState) -> RAGState:
        """
        Add explainability information to the response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with explainability information
        """
        logger.info("Adding explainability information")
        
        try:
            # Check for error from previous steps
            if state.error:
                return state
            
            if self.explainability_tool:
                explainability_data = self.explainability_tool.explain(
                    query=state.query,
                    response=state.response,
                    context_docs=state.context,
                    similarity_scores=state.scores
                )
                
                state.explainability = explainability_data
            else:
                state.explainability = {
                    "context_influence": self._calculate_context_influence(state),
                    "relevant_docs": self._format_relevant_docs(state),
                    "confidence_score": self._calculate_confidence_score(state)
                }
            
            logger.info("Explainability information added")
            
        except Exception as e:
            logger.error(f"Error adding explainability: {e}")
            state.error = f"Error adding explainability: {str(e)}"
        
        return state
    
    def _calculate_context_influence(self, state: RAGState) -> float:
        """Calculate how much the context influenced the response."""
        # Simple heuristic based on similarity scores
        if not state.scores:
            return 0.0
        
        # Average of top 3 similarity scores normalized to 0-1 range
        avg_score = sum(state.scores) / len(state.scores)
        
        # Convert to a percentage (higher is better)
        influence = (1.0 - avg_score) * 100  # Lower distance = higher influence
        
        return min(100.0, max(0.0, influence))  # Clamp to 0-100 range
    
    def _format_relevant_docs(self, state: RAGState) -> List[Dict[str, Any]]:
        """Format the relevant documents for explainability."""
        relevant_docs = []
        
        for i, (doc, score) in enumerate(zip(state.context, state.scores)):
            # Extract relevant metadata
            metadata = doc.metadata.copy() if hasattr(doc, "metadata") else {}
            
            # Make sure we don't include the full content in the metadata
            if "response" in metadata and len(metadata["response"]) > 100:
                metadata["response"] = metadata["response"][:100] + "..."
            
            relevant_docs.append({
                "index": i + 1,
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                "similarity_score": float(score),
                "metadata": metadata
            })
        
        return relevant_docs
    
    def _calculate_confidence_score(self, state: RAGState) -> float:
        """Calculate a confidence score for the generated response."""
        # Simple heuristic combining similarity scores and response length
        if not state.scores or not state.response:
            return 0.0
        
        # 1. Base confidence from similarity scores
        avg_similarity = sum(state.scores) / len(state.scores)
        base_confidence = (1.0 - avg_similarity) * 0.7  # Lower distance = higher confidence
        
        # 2. Response length component (longer responses might indicate more confidence)
        # Normalize to 0-0.3 range with a cap at 250 chars
        response_length = min(len(state.response), 250) / 250.0 * 0.3
        
        # Combine the components
        confidence = (base_confidence + response_length) * 100
        
        return min(100.0, max(0.0, confidence))  # Clamp to 0-100 range
    
    def _store_interaction(self, state: RAGState) -> RAGState:
        """
        Store the interaction in the database.
        
        Args:
            state: Current state
            
        Returns:
            Updated state
        """
        logger.info("Storing interaction in database")
        
        try:
            # Check if database manager is available
            if self.db_manager:
                # Prepare data for storage
                interaction_data = {
                    "query": state.query,
                    "response": state.response,
                    "context_docs": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in state.context
                    ],
                    "similarity_scores": state.scores,
                    "explainability": state.explainability,
                    "error": state.error,
                    "conversation_id": state.conversation_id,
                    "feedback": state.feedback
                }
                
                # Store in database
                interaction_id = self.db_manager.store_interaction(interaction_data)
                logger.info(f"Interaction stored with ID: {interaction_id}")
            else:
                logger.info("Database manager not available, skipping storage")
        
        except Exception as e:
            logger.error(f"Error storing interaction: {e}")
            state.error = f"Error storing interaction: {str(e)}"
        
        return state
    
    def process_query(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User query
            conversation_id: Optional conversation ID for multi-turn conversations
            
        Returns:
            Dictionary with the response and explainability information
        """
        logger.info(f"Processing query: {query}")
        
        # Create initial state
        initial_state = RAGState(
            query=query,
            conversation_id=conversation_id
        )
        
        # Run the graph
        try:
            result_dict = self.graph.invoke(initial_state)
            print("final state")
            print(result_dict)
        
            final_state = RAGState(**result_dict)
            
            result = {
                "query": query,
                "response": final_state.response,
                "error": final_state.error,
                "explainability": final_state.explainability,
                "retrieved_docs_count": len(final_state.context),
                "conversation_id": final_state.conversation_id
            }
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "response": None,
                # "response": f"I'm sorry, there was an error processing your request: {str(e)}",  # A default string response
                "error": f"Error processing query: {str(e)}",
                "explainability": {},
                "retrieved_docs_count": 0,
                "conversation_id": conversation_id
            }
    
    def record_feedback(self, query: str, response: str, feedback: Dict[str, Any]) -> None:
        """
        Record user feedback for a response.
        
        Args:
            query: The original query
            response: The generated response
            feedback: Feedback data (e.g., thumbs_up, rating, comment)
        """
        logger.info(f"Recording feedback for query: {query}")
        
        try:
            if self.db_manager:
                self.db_manager.store_feedback(query, response, feedback)
                logger.info("Feedback recorded successfully")
            else:
                logger.warning("Database manager not available, can't record feedback")
        
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the RAG pipeline with mock components
    from models.embeddings import EmbeddingManager
    from models.llm_manager import LLMManager
    from utils.explainability import ExplainabilityTool
    from database.db_manager import DatabaseManager
    
    # Initialize components
    embedding_manager = EmbeddingManager()
    logger.info("Embedding manager initialized")
    embedding_manager.vector_store = embedding_manager.load_vector_store() # using default faiss
    llm_manager = LLMManager()
    llm_manager.configure_model(model_name=CONFIG.DEFAULT_GROQ_MODEL, provider="groq", api_key="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")
    
    # Create a simple explainability tool
    explainability_tool = ExplainabilityTool()
    
    # for memory
    # db_manager = DatabaseManager(":memory:")
    db_manager = DatabaseManager(CONFIG.DB_PATH)
    
    
    pipeline = RAGPipeline(
        embedding_manager=embedding_manager,
        llm_manager=llm_manager,
        explainability_tool=explainability_tool,
        db_manager=db_manager,
        top_k=3
    )
    
    # Test 
    query = "My laptop arrived with a broken screen. What should I do?"
    
    result = pipeline.process_query(query)
    print(json.dumps(result, indent=2))