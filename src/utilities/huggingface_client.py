# FILE: utilities/huggingface_client.py

import requests
from typing import List, Dict, Optional


class HuggingFaceClient:
    """
    Client for interacting with the Hugging Face API to list and search models.
    """
    
    BASE_URL = "https://huggingface.co/api/models"
    
    def __init__(self):
        self.session = requests.Session()
    
    def list_models(
        self,
        search: Optional[str] = None,
        task: Optional[str] = None,
        limit: int = 20,
        sort: str = "downloads"
    ) -> List[Dict]:
        """
        List models from Hugging Face Hub.
        
        Args:
            search: Search query for model names
            task: Filter by task (e.g., 'image-segmentation', 'object-detection')
            limit: Maximum number of models to return
            sort: Sort order ('downloads', 'likes', 'lastModified')
        
        Returns:
            List of model dictionaries with 'id', 'downloads', 'likes', etc.
        """
        params = {
            "limit": limit,
            "sort": sort,
            "direction": -1  # descending
        }
        
        if search:
            params["search"] = search
        
        if task:
            params["filter"] = task
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch models from Hugging Face: {e}")
    
    def search_models(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search for models by name or description.
        
        Args:
            query: Search query
            limit: Maximum number of models to return
        
        Returns:
            List of model dictionaries
        """
        return self.list_models(search=query, limit=limit)
    
    def get_model_info(self, model_id: str) -> Dict:
        """
        Get detailed information about a specific model.
        
        Args:
            model_id: The model identifier (e.g., 'facebook/detr-resnet-50')
        
        Returns:
            Dictionary with model information
        """
        url = f"{self.BASE_URL}/{model_id}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch model info for {model_id}: {e}")
