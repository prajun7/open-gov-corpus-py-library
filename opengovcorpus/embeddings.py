"""
Embedding generation for different providers
"""

from typing import List, Optional
from abc import ABC, abstractmethod

from .config import load_config
from .exceptions import EmbeddingError


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large"):
        """
        Initialize OpenAI embeddings
        
        Args:
            api_key: OpenAI API key
            model: Model name
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise EmbeddingError("openai package not installed. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}")


class GeminiEmbeddings(EmbeddingProvider):
    """Gemini embedding provider"""
    
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        """
        Initialize Gemini embeddings
        
        Args:
            api_key: Gemini API key
            model: Model name
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise EmbeddingError("google-generativeai package not installed")
        
        genai.configure(api_key=api_key)
        self.model = model
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings"""
        import google.generativeai as genai
        
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Gemini embedding failed: {e}")


class HuggingFaceEmbeddings(EmbeddingProvider):
    """Hugging Face embedding provider"""
    
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize Hugging Face embeddings
        
        Args:
            model: Model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError("sentence-transformers package not installed")
        
        self.model = SentenceTransformer(model)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Hugging Face embedding failed: {e}")


def get_embedding_provider(model: str, api_key: Optional[str] = None) -> EmbeddingProvider:
    """
    Get embedding provider based on model string
    
    Args:
        model: Model string (e.g., "openai/text-embedding-3-large")
        api_key: API key (optional if using config)
        
    Returns:
        EmbeddingProvider instance
    """
    if "/" not in model:
        raise EmbeddingError(f"Invalid model format: {model}. Use 'provider/model-name'")
    
    provider, model_name = model.split("/", 1)
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIEmbeddings(api_key, model_name)
    elif provider == "gemini":
        return GeminiEmbeddings(api_key, model_name)
    elif provider == "hf" or provider == "huggingface":
        return HuggingFaceEmbeddings(model_name)
    else:
        raise EmbeddingError(f"Unknown provider: {provider}")


def create_rag_embeddings(
    model: str,
    vector_db: str = "chroma",
    config_path: Optional[str] = None,
    dataset_path: Optional[str] = None
):
    """
    Create RAG embeddings from dataset
    
    Args:
        model: Model string (e.g., "openai/text-embedding-3-large")
        vector_db: Vector database type
        config_path: Path to config file
        dataset_path: Path to dataset directory
    """
    from .vector_store import create_vector_store
    from .utils import load_from_csv
    from pathlib import Path
    
    print(f"Creating RAG embeddings with model: {model}")
    
    # Load config for API key
    config = load_config(config_path)
    api_key = config.get_api_key()
    
    # Get embedding provider
    provider = get_embedding_provider(model, api_key)
    
    # Find dataset files
    if dataset_path is None:
        # Look for most recent dataset directory
        dataset_dirs = list(Path(".").glob("OpenGovCorpus-*"))
        if not dataset_dirs:
            raise EmbeddingError("No dataset found. Run create_dataset() first.")
        dataset_path = max(dataset_dirs, key=lambda p: p.stat().st_mtime)
    
    dataset_path = Path(dataset_path)
    print(f"Using dataset: {dataset_path}")
    
    # Create vector store
    vector_store = create_vector_store(vector_db, str(dataset_path / "embeddings"))
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        csv_file = dataset_path / f"{split}.csv"
        if not csv_file.exists():
            print(f"Skipping {split} - file not found")
            continue
        
        print(f"Processing {split} split...")
        data = load_from_csv(str(csv_file))
        
        # Skip if no data
        if not data:
            print(f"Skipping {split} - file is empty")
            continue
        
        # Extract texts
        texts = [item['response'] for item in data]
        prompts = [item['prompt'] for item in data]
        
        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]
            batch_data = data[i:i+batch_size]
            
            embeddings = provider.generate_embeddings(batch_texts)
            
            # Add to vector store
            for j, (text, prompt, embedding, metadata) in enumerate(zip(
                batch_texts, batch_prompts, embeddings, batch_data
            )):
                vector_store.add(
                    text=text,
                    embedding=embedding,
                    metadata={
                        'prompt': prompt,
                        'split': split,
                        **metadata
                    },
                    id=f"{split}_{i+j}"
                )
            
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)} items")
        
        print(f"Completed {split} split")
    
    print(f"Embeddings created successfully in: {dataset_path / 'embeddings'}")