
"""
AI Data Manager for Prometheum.

This module provides the main interface for AI-powered data management,
coordinating between storage, indexing, analysis, and query processing.
"""

import logging
import os
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from .analyzer import ContentAnalyzer
from .query import QueryProcessor
from .indexer import ContentIndexer
from .embedding import EmbeddingManager

logger = logging.getLogger(__name__)

class AIDataManager:
    """
    Main interface for AI-powered data management on Prometheum.
    
    This class coordinates between the various AI components:
    - Content indexing: Making files searchable
    - Content analysis: Extracting metadata and insights
    - Query processing: Answering questions about stored data
    """
    
    def __init__(
        self, 
        data_dir: str = "/data",
        index_dir: str = "/var/lib/prometheum/ai/index",
        model_dir: str = "/var/lib/prometheum/ai/models",
        config_path: str = "/var/lib/prometheum/ai/config.json"
    ):
        """
        Initialize the AI Data Manager.
        
        Args:
            data_dir: Base directory for data storage
            index_dir: Directory for storing indices
            model_dir: Directory for AI models
            config_path: Path to AI configuration
        """
        self.data_dir = data_dir
        self.index_dir = index_dir
        self.model_dir = model_dir
        self.config_path = config_path
        
        # Ensure directories exist
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(self.model_dir, self.config)
        self.indexer = ContentIndexer(self.index_dir, self.embedding_manager)
        self.analyzer = ContentAnalyzer(self.model_dir, self.embedding_manager)
        self.query_processor = QueryProcessor(
            self.indexer, 
            self.analyzer,
            self.model_dir,
            self.config
        )
        
        logger.info("AI Data Manager initialized")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load AI configuration from disk.
        
        Returns:
            Dict containing configuration
        """
        default_config = {
            "enabled": True,
            "indexing": {
                "enabled": True,
                "auto_index_new_files": True,
                "supported_extensions": [
                    ".txt", ".md", ".pdf", ".docx", ".html", ".csv", 
                    ".json", ".py", ".js", ".java", ".c", ".cpp", ".h"
                ],
                "excluded_dirs": ["node_modules", "__pycache__", ".git"]
            },
            "analysis": {
                "enabled": True,
                "extract_metadata": True,
                "generate_summaries": True,
                "detect_topics": True,
                "extract_entities": True
            },
            "query": {
                "enabled": True,
                "max_context_size": 4096,
                "max_results": 10,
                "similarity_threshold": 0.7
            },
            "models": {
                "embedding": {
                    "name": "all-MiniLM-L6-v2",
                    "dimensions": 384
                },
                "llm": {
                    "name": "llama-2-7b-q4_0",
                    "context_size": 4096,
                    "system_prompt": "You are a helpful assistant that analyzes and retrieves information from the user's NAS device."
                }
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Update with any missing keys from default
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.error(f"Error loading AI config: {e}")
                return default_config
        else:
            # Create default config
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def index_path(self, path: Union[str, Path], recursive: bool = True) -> Dict[str, Any]:
        """
        Index a file or directory.
        
        Args:
            path: Path to file or directory
            recursive: Whether to index recursively if path is a directory
            
        Returns:
            Dict with indexing results
        """
        path = os.path.abspath(os.path.join(self.data_dir, str(path).lstrip('/')))
        if not os.path.exists(path):
            return {"error": f"Path does not exist: {path}"}
            
        if not self.config["indexing"]["enabled"]:
            return {"error": "Indexing is disabled in configuration"}
            
        logger.info(f"Indexing path: {path}")
        return self.indexer.index_path(path, recursive)
    
    def analyze_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze a file to extract metadata and insights.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with analysis results
        """
        file_path = os.path.abspath(os.path.join(self.data_dir, str(file_path).lstrip('/')))
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return {"error": f"File does not exist: {file_path}"}
            
        if not self.config["analysis"]["enabled"]:
            return {"error": "Analysis is disabled in configuration"}
            
        logger.info(f"Analyzing file: {file_path}")
        return self.analyzer.analyze_file(file_path)
    
    def query(self, query_text: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a natural language query about stored data.
        
        Args:
            query_text: Natural language query
            filters: Optional filters to apply (file types, date ranges, etc.)
            
        Returns:
            Dict with query results
        """
        if not self.config["query"]["enabled"]:
            return {"error": "Query processing is disabled in configuration"}
            
        logger.info(f"Processing query: {query_text}")
        return self.query_processor.process_query(query_text, filters)
    
    def get_catalog_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all indexed and analyzed content.
        
        Returns:
            Dict with catalog summary
        """
        return self.indexer.get_catalog_summary()
    
    def reindex_all(self) -> Dict[str, Any]:
        """
        Rebuild the entire index.
        
        Returns:
            Dict with reindexing results
        """
        logger.info("Rebuilding entire index")
        return self.indexer.rebuild_index(self.data_dir)
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update AI configuration.
        
        Args:
            new_config: New configuration values
            
        Returns:
            Updated configuration
        """
        # Merge with existing config
        for key, value in new_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    # Recursively update nested dicts
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                self.config[key] = value
                
        # Save to disk
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info("Updated AI configuration")
        return self.config

