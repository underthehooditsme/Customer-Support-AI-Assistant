"""FastAPI application for customer support RAG system."""
import logging
from typing import Dict, Any, List, Optional
import os
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import json

# Import from our modules
from config import CONFIG
from models.embeddings import EmbeddingManager
from models.llm_manager import LLMManager
from models.rag_pipeline import RAGPipeline
from database.db_manager import DatabaseManager
from utils.explainability import ExplainabilityTool
from utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the FastAPI app
app = FastAPI(
    title="Customer Support AI Assistant",
    description="AI-powered customer support assistant using RAG",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for multi-turn conversations")
    model_name: Optional[str] = Field(None, description="Optional LLM model name to use")
    model_provider: Optional[str] = Field(None, description="Optional LLM provider (local or groq)")
    api_key: Optional[str] = Field(None, description="Optional API key for external LLM providers")

class FeedbackRequest(BaseModel):
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The generated response")
    rating: Optional[int] = Field(None, description="Optional rating (1-5)")
    is_helpful: Optional[bool] = Field(None, description="Whether the response was helpful")
    comment: Optional[str] = Field(None, description="Optional user comment")
    other_data: Optional[Dict[str, Any]] = Field(None, description="Any other feedback data")

class ConversationRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier")

# Response models
class QueryResponse(BaseModel):
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The generated response")
    error: Optional[str] = Field(None, description="Error message if any")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for multi-turn conversations")
    explainability: Optional[Dict[str, Any]] = Field(None, description="Explainability data")
    retrieved_docs_count: int = Field(0, description="Number of retrieved documents")

class FeedbackResponse(BaseModel):
    success: bool = Field(..., description="Whether the feedback was recorded successfully")
    feedback_id: Optional[str] = Field(None, description="Feedback ID")
    message: str = Field(..., description="Success or error message")

class ConversationResponse(BaseModel):
    conversation_id: str = Field(..., description="Conversation ID")
    message: str = Field(..., description="Success message")

class StatisticsResponse(BaseModel):
    total_interactions: int = Field(0, description="Total number of interactions")
    total_conversations: int = Field(0, description="Total number of conversations")
    total_feedback: int = Field(0, description="Total number of feedback entries")
    average_rating: float = Field(0.0, description="Average feedback rating")
    helpful_responses: int = Field(0, description="Number of responses marked as helpful")
    not_helpful_responses: int = Field(0, description="Number of responses marked as not helpful")
    feedback_rate: float = Field(0.0, description="Percentage of interactions with feedback")

# Global variables for dependency injection
db_manager = None
embedding_manager = None
llm_manager = None
explainability_tool = None
rag_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    global db_manager, embedding_manager, llm_manager, explainability_tool, rag_pipeline
    
    logger.info("Initializing application components")
    
    try:
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(CONFIG.DB_PATH), exist_ok=True)
        
        # Initialize database manager
        db_manager = DatabaseManager(CONFIG.DB_PATH)
        logger.info("Database manager initialized")
        
        # Initialize embedding manager
        # embedding_manager = EmbeddingManager(
        #     model_name=CONFIG.EMBEDDING_MODEL,
        #     vector_db_path=CONFIG.VECTOR_DB_PATH
        # )
        embedding_manager = EmbeddingManager()
        logger.info("Embedding manager initialized")
        embedding_manager.vector_store = embedding_manager.load_vector_store() # using default faiss
        
        # Initialize LLM manager
        # llm_manager = LLMManager(
        #     default_model=CONFIG.DEFAULT_LLM_MODEL,
        #     default_provider=CONFIG.DEFAULT_LLM_PROVIDER
        # )
        llm_manager = LLMManager()
        logger.info("LLM manager initialized")
        
        # Initialize explainability tool
        explainability_tool = ExplainabilityTool()
        logger.info("Explainability tool initialized")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            embedding_manager=embedding_manager,
            llm_manager=llm_manager,
            explainability_tool=explainability_tool,
            db_manager=db_manager,
            top_k=CONFIG.TOP_K
        )
        logger.info("RAG pipeline initialized")
        
    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise

def get_rag_pipeline():
    """Dependency to get the RAG pipeline."""
    return rag_pipeline

def get_db_manager():
    """Dependency to get the database manager."""
    return db_manager

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Customer Support AI Assistant API is running"}

@app.post("/api/generate_response", response_model=QueryResponse, tags=["RAG"])
async def generate_response(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_rag_pipeline)
):
    """
    Generate a response to a customer support query.
    
    Args:
        request: Query request
        background_tasks: FastAPI background tasks
        pipeline: RAG pipeline dependency
        
    Returns:
        Generated response with explainability data
    """
    logger.info(f"Received query: {request.query}")
    
    try:
        # Configure LLM if specified in request
        if request.model_name or request.model_provider or request.api_key:
            pipeline.llm_manager.configure_model(
                model_name=request.model_name,
                provider=request.model_provider,
                api_key=request.api_key
            )
            logger.info(f"Using custom LLM configuration: {request.model_provider}/{request.model_name}")
        
        # Process the query
        result = pipeline.process_query(
            query=request.query,
            conversation_id=request.conversation_id
        )
        
        # Log the interaction asynchronously
        background_tasks.add_task(
            log_interaction,
            query=request.query,
            response=result["response"],
            error=result["error"],
            explainability=result["explainability"],
            conversation_id=result["conversation_id"]
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/api/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    db=Depends(get_db_manager)
):
    """
    Submit feedback for a generated response.
    
    Args:
        request: Feedback request
        db: Database manager dependency
        
    Returns:
        Confirmation of feedback submission
    """
    logger.info(f"Received feedback for query: {request.query[:50]}...")
    
    try:
        # Prepare feedback data
        feedback_data = {
            "rating": request.rating,
            "is_helpful": request.is_helpful,
            "comment": request.comment,
            "other_data": request.other_data or {}
        }
        
        # Store feedback
        feedback_id = db.store_feedback(
            query=request.query,
            response=request.response,
            feedback=feedback_data
        )
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            message="Feedback recorded successfully"
        )
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        return FeedbackResponse(
            success=False,
            feedback_id=None,
            message=f"Error recording feedback: {str(e)}"
        )

@app.post("/api/conversation", response_model=ConversationResponse, tags=["Conversation"])
async def create_conversation(
    request: ConversationRequest,
    db=Depends(get_db_manager)
):
    """
    Create a new conversation.
    
    Args:
        request: Conversation request
        db: Database manager dependency
        
    Returns:
        Conversation ID
    """
    logger.info("Received request to create a new conversation")
    
    try:
        # Create a new conversation
        conversation_id = db.create_conversation(user_id=request.user_id)
        
        return ConversationResponse(
            conversation_id=conversation_id,
            message="Conversation created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")

@app.get("/api/conversation/{conversation_id}", tags=["Conversation"])
async def get_conversation(
    conversation_id: str,
    db=Depends(get_db_manager)
):
    """
    Get the history of a conversation.
    
    Args:
        conversation_id: Conversation ID
        db: Database manager dependency
        
    Returns:
        Conversation history
    """
    logger.info(f"Retrieving history for conversation: {conversation_id}")
    
    try:
        # Get conversation history
        history = db.get_conversation_history(conversation_id=conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@app.delete("/api/conversation/{conversation_id}", tags=["Conversation"])
async def end_conversation(
    conversation_id: str,
    db=Depends(get_db_manager)
):
    """
    End a conversation.
    
    Args:
        conversation_id: Conversation ID
        db: Database manager dependency
        
    Returns:
        Confirmation message
    """
    logger.info(f"Ending conversation: {conversation_id}")
    
    try:
        # End the conversation
        db.end_conversation(conversation_id=conversation_id)
        
        return {
            "conversation_id": conversation_id,
            "message": "Conversation ended successfully"
        }
        
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error ending conversation: {str(e)}")

@app.get("/api/statistics", response_model=StatisticsResponse, tags=["Statistics"])
async def get_statistics(
    db=Depends(get_db_manager)
):
    """
    Get usage statistics.
    
    Args:
        db: Database manager dependency
        
    Returns:
        Usage statistics
    """
    logger.info("Retrieving usage statistics")
    
    try:
        # Get statistics
        statistics = db.get_statistics()
        
        return StatisticsResponse(**statistics)
        
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")

def log_interaction(
    query: str,
    response: str,
    error: Optional[str] = None,
    explainability: Optional[Dict[str, Any]] = None,
    conversation_id: Optional[str] = None
) -> None:
    """
    Log an interaction to the database.
    
    Args:
        query: User query
        response: Generated response
        error: Error message if any
        explainability: Explainability data
        conversation_id: Conversation ID
    """
    logger.info(f"Logging interaction for query: {query[:50]}...")
    
    try:
        # Prepare interaction data
        interaction_data = {
            "query": query,
            "response": response,
            "error": error,
            "explainability": explainability or {},
            "conversation_id": conversation_id
        }
        
        # Store in database
        db_manager.store_interaction(interaction_data)
        
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")

if __name__ == "__main__":
    # Run with Uvicorn
    uvicorn.run(
        "app:app",
        host=CONFIG.HOST,
        port=CONFIG.PORT,
        reload=CONFIG.DEBUG
    )