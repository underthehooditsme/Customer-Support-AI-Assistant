"""Streamlit user interface for the customer support AI assistant."""
import streamlit as st
import requests
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
import uuid
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_MODEL = "mistral-7b"
DEFAULT_PROVIDER = "local"

# Session state initialization
def init_session_state():
    """Initialize session state variables."""
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL
    
    if "model_provider" not in st.session_state:
        st.session_state.model_provider = DEFAULT_PROVIDER
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    
    if "show_explainability" not in st.session_state:
        st.session_state.show_explainability = False
    
    if "current_explainability" not in st.session_state:
        st.session_state.current_explainability = None
    
    if "statistics" not in st.session_state:
        st.session_state.statistics = None

# Helper functions
def create_conversation() -> str:
    """
    Create a new conversation.
    
    Returns:
        Conversation ID
    """
    try:
        response = requests.post(f"{API_URL}/api/conversation", json={})
        data = response.json()
        return data.get("conversation_id")
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        st.error(f"Error creating conversation: {str(e)}")
        return None

def get_response(query: str) -> Dict[str, Any]:
    """
    Get a response from the API.
    
    Args:
        query: User query
        
    Returns:
        API response
    """
    try:
        # Ensure we have a conversation ID
        if not st.session_state.conversation_id:
            st.session_state.conversation_id = create_conversation()
        
        # Request data
        data = {
            "query": query,
            "conversation_id": st.session_state.conversation_id,
            "model_name": st.session_state.model_name,
            "model_provider": st.session_state.model_provider
        }
        
        if st.session_state.api_key:
            data["api_key"] = st.session_state.api_key
        
        # Make the API request
        response = requests.post(f"{API_URL}/api/generate_response", json=data)
        result = response.json()
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting response: {e}")
        return {
            "query": query,
            "response": f"Error: {str(e)}",
            "error": str(e),
            "conversation_id": st.session_state.conversation_id,
            "explainability": None,
            "retrieved_docs_count": 0
        }

def submit_feedback(query: str, response: str, is_helpful: bool, comment: str = None) -> Dict[str, Any]:
    """
    Submit feedback for a response.
    
    Args:
        query: Original query
        response: Generated response
        is_helpful: Whether the response was helpful
        comment: Optional comment
        
    Returns:
        API response
    """
    try:
        # Feedback data
        data = {
            "query": query,
            "response": response,
            "is_helpful": is_helpful,
            "comment": comment
        }
        
        response = requests.post(f"{API_URL}/api/feedback", json=data)
        result = response.json()
        
        return result
    
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return {
            "success": False,
            "message": f"Error submitting feedback: {str(e)}"
        }

def get_statistics() -> Dict[str, Any]:
    """
    Get usage statistics from the API.
    
    Returns:
        Statistics data
    """
    try:
        response = requests.get(f"{API_URL}/api/statistics")
        result = response.json()
        return result
    
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return {
            "total_interactions": 0,
            "total_conversations": 0,
            "total_feedback": 0,
            "average_rating": 0.0,
            "helpful_responses": 0,
            "not_helpful_responses": 0,
            "feedback_rate": 0.0
        }

def end_conversation() -> None:
    """End the current conversation."""
    try:
        if st.session_state.conversation_id:
            requests.delete(f"{API_URL}/api/conversation/{st.session_state.conversation_id}")
            st.session_state.conversation_id = None
            st.session_state.messages = []
            st.success("Conversation ended successfully")
    
    except Exception as e:
        logger.error(f"Error ending conversation: {e}")
        st.error(f"Error ending conversation: {str(e)}")

def display_explainability(explainability: Dict[str, Any]) -> None:
    """
    Display explainability information.
    
    Args:
        explainability: Explainability data
    """
    if not explainability:
        st.info("No explainability data available")
        return
    
    st.subheader("Response Explainability")
    
    tab1, tab2, tab3 = st.tabs(["Confidence", "Context Influence", "Retrieved Documents"])
    
    with tab1:

        confidence_score = explainability.get("confidence_score", 0)
        st.metric("Confidence Score", f"{confidence_score:.2f}%")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:

        context_influence = explainability.get("context_influence", 0)
        st.metric("Context Influence", f"{context_influence:.2f}%")

        st.progress(context_influence / 100.0)
        
        st.write("This indicates how much the retrieved documents influenced the final response.")
    
    with tab3:

        relevant_docs = explainability.get("context_documents", [])
        
        if not relevant_docs:
            st.info("No retrieved documents information available")
            return
        
        doc_data = []
        for doc in relevant_docs:
            doc_data.append({
                "Index": doc.get("index", 0),
                "Similarity": f"{(1 - doc.get('relevance_score', 0)) * 100:.2f}%",
                "Content Preview": doc.get("content_preview", "N/A")
            })
        
        if doc_data:
            st.dataframe(pd.DataFrame(doc_data), use_container_width=True)

def display_statistics(statistics: Dict[str, Any]) -> None:
    """
    Display usage statistics.
    
    Args:
        statistics: Statistics data
    """
    st.subheader("📈 System Statistics")
    

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Interactions", statistics.get("total_interactions", 0))
    with col2:
        st.metric("Total Conversations", statistics.get("total_conversations", 0))
    with col3:
        st.metric("Feedback Rate", f"{statistics.get('feedback_rate', 0):.2f}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Rating", f"{statistics.get('average_rating', 0):.2f}/5")
    with col2:
        st.metric("Helpful Responses", statistics.get("helpful_responses", 0))
    with col3:
        st.metric("Not Helpful Responses", statistics.get("not_helpful_responses", 0))
    
    helpful = statistics.get("helpful_responses", 0)
    not_helpful = statistics.get("not_helpful_responses", 0)
    
    if helpful + not_helpful > 0:
        fig = px.pie(
            names=["Helpful", "Not Helpful"],
            values=[helpful, not_helpful],
            title="Response Helpfulness",
            color_discrete_sequence=["green", "red"]
        )
        st.plotly_chart(fig, use_container_width=True)

# UI Components
def model_selection_sidebar():
    """Sidebar for model selection and API key."""
    st.sidebar.title("Settings")
    
    # Model provider selection
    st.sidebar.subheader("Model Provider")
    provider_options = ["local", "groq"]
    st.session_state.model_provider = st.sidebar.selectbox(
        "Select provider:",
        options=provider_options,
        index=provider_options.index(st.session_state.model_provider)
    )
    
    # Model selection based on provider
    st.sidebar.subheader("Model Selection")
    
    if st.session_state.model_provider == "local":
        model_options = ["meta-llama/Meta-Llama-3-8B"]
        st.session_state.model_name = st.sidebar.selectbox(
            "Select local model:",
            options=model_options,
            index=0 if st.session_state.model_name not in model_options else model_options.index(st.session_state.model_name)
        )
    else:  # groq
        model_options = ["Llama3-70b-8192", "Llama3-8b-8192", "Mistral-Saba-24b"]
        st.session_state.model_name = st.sidebar.selectbox(
            "Select Groq model:",
            options=model_options,
            index=0 if st.session_state.model_name not in model_options else model_options.index(st.session_state.model_name)
        )
        
        # API key for Groq
        st.sidebar.subheader("API Key")
        st.session_state.api_key = st.sidebar.text_input(
            "Enter Groq API key:",
            value=st.session_state.api_key,
            type="password"
        )
    
    # Explainability toggle
    st.sidebar.subheader("Explainability")
    st.session_state.show_explainability = st.sidebar.toggle(
        "Show explainability",
        value=st.session_state.show_explainability
    )
    
    # New conversation button
    st.sidebar.subheader("Conversation")
    if st.sidebar.button("New Conversation"):
        end_conversation()
        st.session_state.conversation_id = create_conversation()
        st.rerun()
    
    # Statistics button
    st.sidebar.subheader("System Statistics")
    if st.sidebar.button("View Statistics"):
        st.session_state.statistics = get_statistics()
        st.rerun()

def message_area():
    """Display the chat message area."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Add feedback buttons for assistant messages
            if message["role"] == "assistant" and "feedback_submitted" not in message:
                col1, col2, col3 = st.columns([1, 1, 5])
                with col1:
                    if st.button("👍", key=f"thumbs_up_{message['id']}"):
                        query = next((m["content"] for m in st.session_state.messages if m["id"] == message["query_id"]), "")
                        result = submit_feedback(query, message["content"], True)
                        if result.get("success", False):
                            message["feedback_submitted"] = True
                            st.success("Thank you for your feedback!")
                            st.rerun()
                        else:
                            st.error(f"Error submitting feedback: {result.get('message', 'Unknown error')}")
                
                with col2:
                    if st.button("👎", key=f"thumbs_down_{message['id']}"):
                        query = next((m["content"] for m in st.session_state.messages if m["id"] == message["query_id"]), "")
                        result = submit_feedback(query, message["content"], False)
                        if result.get("success", False):
                            message["feedback_submitted"] = True
                            st.success("Thank you for your feedback!")
                            st.rerun()
                        else:
                            st.error(f"Error submitting feedback: {result.get('message', 'Unknown error')}")
                
                with col3:
                    if st.button("Why this response?", key=f"why_{message['id']}"):
                        st.session_state.current_explainability = message.get("explainability", {})
                        st.rerun()

def chat_interface():
    """Main chat interface."""
    st.title("Customer Support AI Assistant")
    
    # Initialize session state
    init_session_state()
    
    # Show sidebar
    model_selection_sidebar()
    
    # Show message area
    message_area()
    
    # Show explainability if requested
    if st.session_state.show_explainability and st.session_state.current_explainability:
        display_explainability(st.session_state.current_explainability)
        if st.button("Hide Explainability"):
            st.session_state.current_explainability = None
            st.rerun()
    
    # Show statistics if requested
    if st.session_state.statistics:
        display_statistics(st.session_state.statistics)
        if st.button("Hide Statistics"):
            st.session_state.statistics = None
            st.rerun()
    
    # Chat input
    if query := st.chat_input("How can I help you today?"):
        # Ensure we have a conversation ID
        if not st.session_state.conversation_id:
            st.session_state.conversation_id = create_conversation()
        
        # Add user message
        user_message_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "id": user_message_id
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get response
        with st.spinner("Thinking..."):
            result = get_response(query)
        
        # Add assistant message
        assistant_message_id = str(uuid.uuid4())
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "id": assistant_message_id,
            "query_id": user_message_id,
            "explainability": result.get("explainability", {})
        })
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(result["response"])
            
            # Add feedback buttons
            col1, col2, col3 = st.columns([1, 1, 5])
            with col1:
                if st.button("👍", key=f"thumbs_up_{assistant_message_id}"):
                    feedback_result = submit_feedback(query, result["response"], True)
                    if feedback_result.get("success", False):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error(f"Error submitting feedback: {feedback_result.get('message', 'Unknown error')}")
            
            with col2:
                if st.button("👎", key=f"thumbs_down_{assistant_message_id}"):
                    feedback_result = submit_feedback(query, result["response"], False)
                    if feedback_result.get("success", False):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error(f"Error submitting feedback: {feedback_result.get('message', 'Unknown error')}")
            
            with col3:
                if st.button("Why this response?", key=f"why_{assistant_message_id}"):
                    st.session_state.current_explainability = result.get("explainability", {})
                    st.rerun()

if __name__ == "__main__":
    chat_interface()
