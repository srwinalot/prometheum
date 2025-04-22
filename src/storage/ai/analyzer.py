
"""
Content Analyzer for Prometheum.

This module handles the analysis of file content, extracting metadata,
summaries, topics, and other insights using LLMs.
"""

import logging
import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import hashlib
import mimetypes

from .embedding import EmbeddingManager
from ..utils import file_utils

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """
    Analyzes file content to extract insights using AI models.
    
    Capabilities:
    - File metadata extraction
    - Content summarization
    - Topic detection
    - Entity extraction
    - Document classification
    """
    
    def __init__(
        self, 
        model_dir: str,
        embedding_manager: EmbeddingManager,
        analysis_cache_dir: str = "/var/lib/prometheum/ai/analysis_cache"
    ):
        """
        Initialize the content analyzer.
        
        Args:
            model_dir: Directory containing AI models
            embedding_manager: For generating embeddings
            analysis_cache_dir: Directory to cache analysis results
        """
        self.model_dir = model_dir
        self.embedding_manager = embedding_manager
        self.analysis_cache_dir = analysis_cache_dir
        self.llm = None  # Lazy-loaded
        
        # Ensure cache directory exists
        os.makedirs(self.analysis_cache_dir, exist_ok=True)
        
        # Initialize file type handlers
        self._init_file_handlers()
        
        logger.info("Content Analyzer initialized")
    
    def _init_file_handlers(self):
        """Initialize handlers for different file types."""
        # Map file extensions to handler functions
        self.file_handlers = {
            ".txt": self._analyze_text,
            ".md": self._analyze_text,
            ".pdf": self._analyze_pdf,
            ".docx": self._analyze_docx,
            ".xlsx": self._analyze_xlsx,
            ".csv": self._analyze_csv,
            ".json": self._analyze_json,
            ".py": self._analyze_code,
            ".js": self._analyze_code,
            ".html": self._analyze_html,
            ".jpg": self._analyze_image,
            ".jpeg": self._analyze_image,
            ".png": self._analyze_image,
            ".mp3": self._analyze_audio,
            ".mp4": self._analyze_video,
        }
    
    def _get_llm(self):
        """Lazy-load the LLM model."""
        if self.llm is None:
            try:
                from llama_cpp import Llama
                
                model_path = os.path.join(self.model_dir, "llama-2-7b-q4_0.gguf")
                
                # If model doesn't exist, we'd download it here in a real implementation
                if not os.path.exists(model_path):
                    logger.warning(f"LLM model not found at {model_path}")
                    # In a real implementation, we would download the model here
                    # For now, we'll just create a placeholder file
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    with open(model_path, 'w') as f:
                        f.write("# Placeholder for LLM model\n")
                
                # Initialize the LLM
                self.llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=4
                )
                logger.info("LLM loaded successfully")
            except Exception as e:
                logger.error(f"Error loading LLM: {e}")
                # Fallback to dummy LLM for development
                self.llm = DummyLLM()
        
        return self.llm
    
    def analyze_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze a file to extract metadata and insights.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with analysis results
        """
        file_path = str(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            return {"error": f"File does not exist: {file_path}"}
        
        # Check cache first
        cache_key = self._generate_cache_key(file_path)
        cached_analysis = self._get_cached_analysis(cache_key)
        if cached_analysis:
            return cached_analysis
        
        # Get file info
        file_ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        file_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        
        # Basic metadata
        analysis = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "file_extension": file_ext,
            "file_size": file_size,
            "file_type": file_type,
            "last_modified": os.path.getmtime(file_path),
            "created": os.path.getctime(file_path),
            "analysis_time": time.time(),

