import logging
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
import json
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for customer support interactions."""
    
    def __init__(self, db_path: str):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database or ":memory:" for in-memory database
        """
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        logger.info(f"Setting up database at {self.db_path}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create interactions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                query TEXT,
                response TEXT,
                context_docs TEXT,
                similarity_scores TEXT,
                explainability TEXT,
                error TEXT,
                conversation_id TEXT,
                feedback_id TEXT
            )
            ''')
            
            # Create feedback table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                interaction_id TEXT,
                timestamp TEXT,
                rating INTEGER,
                comment TEXT,
                is_helpful BOOLEAN,
                other_data TEXT,
                FOREIGN KEY (interaction_id) REFERENCES interactions(id)
            )
            ''')
            
            # Create conversations table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                start_timestamp TEXT,
                end_timestamp TEXT,
                user_id TEXT,
                session_data TEXT
            )
            ''')
            
            conn.commit()
            logger.info("Database tables created successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        
        finally:
            if conn:
                conn.close()
    
    def store_interaction(self, interaction_data: Dict[str, Any]) -> str:
        """
        Store an interaction in the database.
        
        Args:
            interaction_data: Dictionary containing interaction data
            
        Returns:
            Interaction ID
        """
        logger.info("Storing interaction in database")
        
        try:
            # Generate a unique ID
            interaction_id = str(uuid.uuid4())
            
            # Create a connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract data
            query = interaction_data.get("query", "")
            response = interaction_data.get("response", "")
            context_docs = json.dumps(interaction_data.get("context_docs", []))
            similarity_scores = json.dumps(interaction_data.get("similarity_scores", []))
            explainability = json.dumps(interaction_data.get("explainability", {}))
            error = interaction_data.get("error", None)
            conversation_id = interaction_data.get("conversation_id", None)
            feedback_id = interaction_data.get("feedback_id", None)
            
            # Current timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert into database
            cursor.execute('''
            INSERT INTO interactions 
            (id, timestamp, query, response, context_docs, similarity_scores, explainability, 
             error, conversation_id, feedback_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                interaction_id, 
                timestamp, 
                query, 
                response, 
                context_docs, 
                similarity_scores, 
                explainability, 
                error, 
                conversation_id, 
                feedback_id
            ))
            
            # Commit the transaction
            conn.commit()
            logger.info(f"Interaction stored with ID: {interaction_id}")
            
            return interaction_id
            
        except sqlite3.Error as e:
            logger.error(f"Error storing interaction: {e}")
            raise
        
        finally:
            if conn:
                conn.close()
    
    def store_feedback(self, query: str, response: str, feedback: Dict[str, Any]) -> str:
        """
        Store user feedback for a response.
        
        Args:
            query: The original query
            response: The generated response
            feedback: Feedback data (e.g., rating, comment, is_helpful)
            
        Returns:
            Feedback ID
        """
        logger.info(f"Storing feedback for query: {query[:50]}...")
        
        try:
            # Generate a unique ID
            feedback_id = str(uuid.uuid4())
            
            # Find the interaction ID for this query/response pair
            interaction_id = self._find_interaction_id(query, response)
            
            # Create a connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract feedback data
            rating = feedback.get("rating", None)
            comment = feedback.get("comment", None)
            is_helpful = feedback.get("is_helpful", None)
            other_data = json.dumps(feedback.get("other_data", {}))
            
            # Current timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert into database
            cursor.execute('''
            INSERT INTO feedback 
            (id, interaction_id, timestamp, rating, comment, is_helpful, other_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_id, 
                interaction_id, 
                timestamp, 
                rating, 
                comment, 
                is_helpful, 
                other_data
            ))
            
            # Update interaction with the feedback ID
            if interaction_id:
                cursor.execute('''
                UPDATE interactions SET feedback_id = ? WHERE id = ?
                ''', (feedback_id, interaction_id))
            
            # Commit the transaction
            conn.commit()
            logger.info(f"Feedback stored with ID: {feedback_id}")
            
            return feedback_id
            
        except sqlite3.Error as e:
            logger.error(f"Error storing feedback: {e}")
            raise
        
        finally:
            if conn:
                conn.close()
    
    def _find_interaction_id(self, query: str, response: str) -> Optional[str]:
        """
        Find the interaction ID for a given query/response pair.
        
        Args:
            query: The original query
            response: The generated response
            
        Returns:
            Interaction ID if found, None otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query the database
            cursor.execute('''
            SELECT id FROM interactions 
            WHERE query = ? AND response = ?
            ORDER BY timestamp DESC
            LIMIT 1
            ''', (query, response))
            
            result = cursor.fetchone()
            
            return result[0] if result else None
            
        except sqlite3.Error as e:
            logger.error(f"Error finding interaction: {e}")
            return None
        
        finally:
            if conn:
                conn.close()
    
    def create_conversation(self, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Conversation ID
        """
        logger.info("Creating new conversation")
        
        try:
            # Generate a unique ID
            conversation_id = str(uuid.uuid4())
            
            # Create a connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Current timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert into database
            cursor.execute('''
            INSERT INTO conversations 
            (id, start_timestamp, user_id, session_data)
            VALUES (?, ?, ?, ?)
            ''', (
                conversation_id, 
                timestamp, 
                user_id, 
                '{}'
            ))
            
            # Commit the transaction
            conn.commit()
            logger.info(f"Conversation created with ID: {conversation_id}")
            
            return conversation_id
            
        except sqlite3.Error as e:
            logger.error(f"Error creating conversation: {e}")
            raise
        
        finally:
            if conn:
                conn.close()
    
    def end_conversation(self, conversation_id: str) -> None:
        """
        Mark a conversation as ended.
        
        Args:
            conversation_id: Conversation ID to end
        """
        logger.info(f"Ending conversation: {conversation_id}")
        
        try:
            # Create a connection
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Current timestamp
            timestamp = datetime.now().isoformat()
            
            # Update the conversation
            cursor.execute('''
            UPDATE conversations SET end_timestamp = ? WHERE id = ?
            ''', (timestamp, conversation_id))
            
            # Commit the transaction
            conn.commit()
            logger.info(f"Conversation {conversation_id} marked as ended")
            
        except sqlite3.Error as e:
            logger.error(f"Error ending conversation: {e}")
            raise
        
        finally:
            if conn:
                conn.close()
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get the history of a conversation.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            List of interactions in the conversation
        """
        logger.info(f"Retrieving history for conversation: {conversation_id}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Query the database
            cursor.execute('''
            SELECT id, timestamp, query, response
            FROM interactions 
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
            ''', (conversation_id,))
            
            # Fetch all results
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            history = []
            for row in results:
                history.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "query": row[2],
                    "response": row[3]
                })
            
            logger.info(f"Retrieved {len(history)} interactions for conversation {conversation_id}")
            
            return history
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving conversation history: {e}")
            return []
        
        finally:
            if conn:
                conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics from the database.
        
        Returns:
            Dictionary with statistics
        """
        logger.info("Retrieving usage statistics")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total interactions
            cursor.execute("SELECT COUNT(*) FROM interactions")
            total_interactions = cursor.fetchone()[0]
            
            # Total conversations
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_conversations = cursor.fetchone()[0]
            
            # Total feedback
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]
            
            # Average feedback rating
            cursor.execute("SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL")
            avg_rating = cursor.fetchone()[0] or 0
            
            # Helpful responses count
            cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_helpful = 1")
            helpful_count = cursor.fetchone()[0]
            
            # Not helpful responses count
            cursor.execute("SELECT COUNT(*) FROM feedback WHERE is_helpful = 0")
            not_helpful_count = cursor.fetchone()[0]
            
            # Construct statistics dictionary
            statistics = {
                "total_interactions": total_interactions,
                "total_conversations": total_conversations,
                "total_feedback": total_feedback,
                "average_rating": round(float(avg_rating), 2),
                "helpful_responses": helpful_count,
                "not_helpful_responses": not_helpful_count,
                "feedback_rate": round(total_feedback / total_interactions * 100, 2) if total_interactions > 0 else 0
            }
            
            logger.info("Retrieved usage statistics successfully")
            
            return statistics
            
        except sqlite3.Error as e:
            logger.error(f"Error retrieving statistics: {e}")
            return {}
        
        finally:
            if conn:
                conn.close()
