"""Logging utilities for the customer support RAG system."""
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Define log levels mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

def setup_logging(
    name: str,
    log_level: str = "info",
    log_file: Optional[str] = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        log_level: Logging level (debug, info, warning, error, critical)
        log_file: Optional file path to write logs to
        log_format: Format string for log messages
        
    Returns:
        Configured logger
    """
    # Get the logger
    logger = logging.getLogger(name)
    
    # Set the log level
    level = LOG_LEVELS.get(log_level.lower(), logging.INFO)
    logger.setLevel(level)
    
    # Create a formatter
    formatter = logging.Formatter(log_format)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create the file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Add handlers to the logger
    for handler in handlers:
        logger.addHandler(handler)
    
    return logger

class QueryLogger:
    """Logger specifically for user queries and responses."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the query logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = log_dir
        
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.logger = logging.getLogger("query_logger")
        self.logger.setLevel(logging.INFO)
        
        query_log_file = os.path.join(log_dir, "queries.log")
        query_handler = logging.FileHandler(query_log_file)
        query_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(query_handler)
    
    def log_query(self, query: str, response: str, metadata: Optional[dict] = None) -> None:
        """
        Log a query and its response.
        
        Args:
            query: User query
            response: Generated response
            metadata: Optional metadata about the query/response
        """
        # Construct the log message
        log_message = f"QUERY: {query}\nRESPONSE: {response}"
        
        # Add metadata if provided
        if metadata:
            log_message += f"\nMETADATA: {metadata}"
        
        # Log the message
        self.logger.info(log_message)
    
    def log_error(self, query: str, error: str) -> None:
        """
        Log an error that occurred while processing a query.
        
        Args:
            query: User query
            error: Error message
        """
        self.logger.error(f"ERROR PROCESSING QUERY: {query}\nERROR: {error}")

class PerformanceLogger:
    """Logger for tracking system performance."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the performance logger.
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = log_dir
        
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create a logger
        self.logger = logging.getLogger("performance_logger")
        self.logger.setLevel(logging.INFO)
        
        # Create a handler for performance logs
        perf_log_file = os.path.join(log_dir, "performance.log")
        perf_handler = logging.FileHandler(perf_log_file)
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(perf_handler)
    
    def log_retrieval_performance(
        self,
        query: str,
        num_documents: int,
        time_taken_ms: float,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Log the performance of document retrieval.
        
        Args:
            query: User query
            num_documents: Number of documents retrieved
            time_taken_ms: Time taken in milliseconds
            metadata: Optional additional metadata
        """
        # Construct the log message
        log_message = f"RETRIEVAL - Query: {query[:50]}... | Docs: {num_documents} | Time: {time_taken_ms:.2f}ms"
        
        # Add metadata if provided
        if metadata:
            log_message += f" | Metadata: {metadata}"
        
        # Log the message
        self.logger.info(log_message)
    
    def log_generation_performance(
        self,
        query: str,
        response_length: int,
        time_taken_ms: float,
        model: str,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Log the performance of response generation.
        
        Args:
            query: User query
            response_length: Length of the generated response
            time_taken_ms: Time taken in milliseconds
            model: Model used for generation
            metadata: Optional additional metadata
        """
        # Construct the log message
        log_message = (
            f"GENERATION - Query: {query[:50]}... | "
            f"Response Length: {response_length} | "
            f"Time: {time_taken_ms:.2f}ms | "
            f"Model: {model}"
        )
        
        if metadata:
            log_message += f" | Metadata: {metadata}"
        
        self.logger.info(log_message)
    
    def log_system_metrics(self, metrics: dict) -> None:
        """
        Log system metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        metrics_str = ", ".join(f"{key}={value}" for key, value in metrics.items())
        self.logger.info(f"SYSTEM METRICS: {metrics_str}")