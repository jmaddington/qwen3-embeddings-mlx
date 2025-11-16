#!/usr/bin/env python3
"""
API Tests for Qwen3 Embedding Server

Run with: python tests/test_api.py
Or with pytest: pytest tests/test_api.py -v
"""

import sys
import time
import json
import base64
import struct
from typing import List, Dict, Any
import requests
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"
TOLERANCE = 0.01  # For float comparisons

# Model configurations
MODELS = {
    "small": {"dim": 1024, "alias": "small"},
    "medium": {"dim": 2560, "alias": "medium"},
    "large": {"dim": 4096, "alias": "large"}
}

# Default dimension for backward compatibility
EMBEDDING_DIM = 1024

class TestClient:
    """Test client for Qwen3 Embedding Server"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def check_server(self) -> bool:
        """Check if server is running"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def test_health(self) -> Dict[str, Any]:
        """Test health endpoint"""
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200, f"Health check failed: {response.status_code}"
        
        data = response.json()
        assert "status" in data
        assert "model_status" in data
        assert "embedding_dim" in data
        # Default model should be small (1024 dim)
        assert data["embedding_dim"] == 1024
        
        return data
    
    def test_models_endpoint(self) -> Dict[str, Any]:
        """Test models listing endpoint"""
        response = self.session.get(f"{self.base_url}/models")
        assert response.status_code == 200, f"Models endpoint failed: {response.status_code}"
        
        data = response.json()
        assert "models" in data
        assert "default_model" in data
        assert "loaded_models" in data
        
        # Check that all expected models are listed
        models = data["models"]
        assert len(models) >= 3, "Should have at least 3 models available"
        
        return data
    
    def test_single_embedding(self, model: str = None) -> Dict[str, Any]:
        """Test single text embedding"""
        test_text = "Machine learning is transforming the world"
        
        payload = {"text": test_text, "normalize": True}
        if model:
            payload["model"] = model
        
        response = self.session.post(
            f"{self.base_url}/embed",
            json=payload
        )
        
        assert response.status_code == 200, f"Embedding failed: {response.text}"
        
        data = response.json()
        assert "embedding" in data
        assert "dim" in data
        assert "normalized" in data
        assert "processing_time_ms" in data
        
        # Get expected dimension for model
        expected_dim = MODELS.get(model, {"dim": 1024})["dim"] if model else 1024
        
        # Validate embedding
        embedding = np.array(data["embedding"])
        assert embedding.shape == (expected_dim,), f"Wrong dimension: {embedding.shape} (expected {expected_dim})"
        assert data["dim"] == expected_dim, f"Dimension mismatch in response"
        
        # Check normalization
        if data["normalized"]:
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < TOLERANCE, f"Not normalized: norm={norm}"
        
        return data
    
    def test_batch_embedding(self, model: str = None) -> Dict[str, Any]:
        """Test batch embedding"""
        test_texts = [
            "Python is a great programming language",
            "FastAPI makes building APIs easy",
            "MLX is optimized for Apple Silicon"
        ]
        
        payload = {"texts": test_texts, "normalize": True}
        if model:
            payload["model"] = model
        
        response = self.session.post(
            f"{self.base_url}/embed_batch",
            json=payload
        )
        
        assert response.status_code == 200, f"Batch embedding failed: {response.text}"
        
        data = response.json()
        assert "embeddings" in data
        assert "count" in data
        assert "dim" in data
        assert "normalized" in data
        
        # Get expected dimension for model
        expected_dim = MODELS.get(model, {"dim": 1024})["dim"] if model else 1024
        
        # Validate embeddings
        embeddings = np.array(data["embeddings"])
        assert embeddings.shape == (len(test_texts), expected_dim), f"Wrong shape: {embeddings.shape}"
        assert data["dim"] == expected_dim, f"Dimension mismatch in response"
        assert data["count"] == len(test_texts)
        
        # Check normalization
        if data["normalized"]:
            for emb in embeddings:
                norm = np.linalg.norm(emb)
                assert abs(norm - 1.0) < TOLERANCE, f"Not normalized: norm={norm}"
        
        return data
    
    def test_empty_text(self) -> None:
        """Test handling of empty text"""
        response = self.session.post(
            f"{self.base_url}/embed",
            json={"text": ""}
        )
        
        assert response.status_code == 422, "Empty text should be rejected"
    
    def test_large_batch(self) -> None:
        """Test handling of large batch"""
        large_texts = ["Test text"] * 100  # Exceeds typical max_batch_size
        
        response = self.session.post(
            f"{self.base_url}/embed_batch",
            json={"texts": large_texts}
        )
        
        # Should either succeed or return 422 for exceeding limit
        assert response.status_code in [200, 422]
    
    def test_similarity(self) -> None:
        """Test semantic similarity"""
        pairs = [
            ("dog", "puppy", 0.3),  # Should be similar
            ("dog", "car", 0.1),     # Should be dissimilar
            ("AI", "artificial intelligence", 0.2),  # Should be similar
        ]
        
        for text1, text2, min_similarity in pairs:
            response = self.session.post(
                f"{self.base_url}/embed_batch",
                json={"texts": [text1, text2]}
            )
            
            assert response.status_code == 200
            embeddings = np.array(response.json()["embeddings"])
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1])
            
            if min_similarity > 0:
                assert similarity >= min_similarity, \
                    f"'{text1}' and '{text2}' similarity {similarity:.3f} < {min_similarity}"
    
    def test_openai_single_embedding(self, model: str = "small") -> Dict[str, Any]:
        """Test OpenAI-compatible endpoint with single text input"""
        test_text = "OpenAI compatible embedding test"

        payload = {
            "input": test_text,
            "model": model,
            "encoding_format": "float"
        }

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        )

        assert response.status_code == 200, f"OpenAI embedding failed: {response.text}"

        data = response.json()
        assert data["object"] == "list", "Response object should be 'list'"
        assert "data" in data, "Missing 'data' field"
        assert "model" in data, "Missing 'model' field"
        assert "usage" in data, "Missing 'usage' field"

        # Validate embedding data
        assert len(data["data"]) == 1, "Should have one embedding"
        embedding_obj = data["data"][0]
        assert embedding_obj["object"] == "embedding"
        assert embedding_obj["index"] == 0
        assert isinstance(embedding_obj["embedding"], list)

        # Get expected dimension for model
        expected_dim = MODELS.get(model, {"dim": 1024})["dim"]
        embedding = np.array(embedding_obj["embedding"])
        assert embedding.shape == (expected_dim,), f"Wrong dimension: {embedding.shape}"

        # Check normalization (OpenAI always normalizes)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < TOLERANCE, f"Not normalized: norm={norm}"

        # Validate usage
        usage = data["usage"]
        assert "prompt_tokens" in usage
        assert "total_tokens" in usage
        assert usage["prompt_tokens"] > 0, "Token count should be > 0"
        assert usage["total_tokens"] == usage["prompt_tokens"]

        return data

    def test_openai_batch_embedding(self, model: str = "small") -> Dict[str, Any]:
        """Test OpenAI-compatible endpoint with array of texts"""
        test_texts = [
            "First test sentence for OpenAI",
            "Second test sentence",
            "Third test sentence"
        ]

        payload = {
            "input": test_texts,
            "model": model,
            "encoding_format": "float"
        }

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        )

        assert response.status_code == 200, f"OpenAI batch embedding failed: {response.text}"

        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == len(test_texts), "Wrong number of embeddings"

        # Get expected dimension for model
        expected_dim = MODELS.get(model, {"dim": 1024})["dim"]

        # Validate each embedding
        for idx, embedding_obj in enumerate(data["data"]):
            assert embedding_obj["object"] == "embedding"
            assert embedding_obj["index"] == idx
            assert isinstance(embedding_obj["embedding"], list)

            embedding = np.array(embedding_obj["embedding"])
            assert embedding.shape == (expected_dim,), f"Wrong dimension at index {idx}"

            # Check normalization
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < TOLERANCE, f"Not normalized at index {idx}: norm={norm}"

        # Validate usage
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"]

        return data

    def test_openai_base64_encoding(self, model: str = "small") -> None:
        """Test OpenAI endpoint with base64 encoding format"""
        test_text = "Base64 encoding test"

        payload = {
            "input": test_text,
            "model": model,
            "encoding_format": "base64"
        }

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        )

        assert response.status_code == 200, f"Base64 encoding failed: {response.text}"

        data = response.json()
        embedding_obj = data["data"][0]

        # Embedding should be a base64 string
        assert isinstance(embedding_obj["embedding"], str), "Embedding should be base64 string"

        # Decode base64 to verify it's valid
        try:
            b64_bytes = base64.b64decode(embedding_obj["embedding"])
            # Get expected dimension for model
            expected_dim = MODELS.get(model, {"dim": 1024})["dim"]
            expected_bytes = expected_dim * 4  # 4 bytes per float32
            assert len(b64_bytes) == expected_bytes, f"Wrong byte length: {len(b64_bytes)}"

            # Unpack floats
            floats = struct.unpack(f'<{expected_dim}f', b64_bytes)
            embedding = np.array(floats)

            # Check normalization
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < TOLERANCE, f"Not normalized: norm={norm}"

        except Exception as e:
            raise AssertionError(f"Failed to decode base64 embedding: {e}")

    def test_openai_invalid_model(self) -> None:
        """Test OpenAI endpoint with invalid model name"""
        payload = {
            "input": "test",
            "model": "nonexistent-model"
        }

        response = self.session.post(
            f"{self.base_url}/v1/embeddings",
            json=payload
        )

        # Should return error status
        assert response.status_code in [400, 500], "Should reject invalid model"

    def test_performance(self) -> Dict[str, float]:
        """Test performance metrics"""
        metrics = {}

        # Single embedding latency
        times = []
        for _ in range(5):
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/embed",
                json={"text": "Performance test"}
            )
            times.append((time.time() - start) * 1000)
            assert response.status_code == 200

        metrics["single_embed_ms"] = np.mean(times[1:])  # Skip first (warmup)

        # Batch embedding latency
        times = []
        for _ in range(3):
            start = time.time()
            response = self.session.post(
                f"{self.base_url}/embed_batch",
                json={"texts": ["Test"] * 10}
            )
            times.append((time.time() - start) * 1000)
            assert response.status_code == 200

        metrics["batch_10_ms"] = np.mean(times)
        metrics["throughput_per_sec"] = 10000 / metrics["batch_10_ms"]

        return metrics

def run_tests():
    """Run all tests"""
    print("🧪 Qwen3 Embedding Server - Test Suite")
    print("=" * 50)
    
    client = TestClient()
    
    # Check server
    if not client.check_server():
        print("❌ Server is not running. Start with: python server.py")
        return False
    
    results = {"passed": 0, "failed": 0}
    
    # Test suite
    tests = [
        ("Health Check", client.test_health),
        ("Models Endpoint", client.test_models_endpoint),
        ("Single Embedding (default)", lambda: client.test_single_embedding()),
        ("Single Embedding (small)", lambda: client.test_single_embedding("small")),
        ("Single Embedding (medium)", lambda: client.test_single_embedding("medium")),
        ("Batch Embedding (default)", lambda: client.test_batch_embedding()),
        ("Batch Embedding (medium)", lambda: client.test_batch_embedding("medium")),
        ("Empty Text Validation", client.test_empty_text),
        ("Large Batch Handling", client.test_large_batch),
        ("Semantic Similarity", client.test_similarity),
        ("OpenAI Single Embedding (small)", lambda: client.test_openai_single_embedding("small")),
        ("OpenAI Single Embedding (medium)", lambda: client.test_openai_single_embedding("medium")),
        ("OpenAI Batch Embedding (small)", lambda: client.test_openai_batch_embedding("small")),
        ("OpenAI Base64 Encoding", lambda: client.test_openai_base64_encoding("small")),
        ("OpenAI Invalid Model", client.test_openai_invalid_model),
        ("Performance Metrics", client.test_performance),
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 {test_name}")
            result = test_func()
            
            if result:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, float):
                            print(f"  ✓ {key}: {value:.2f}")
                        else:
                            print(f"  ✓ {key}: {value}")
            
            print(f"  ✅ Passed")
            results["passed"] += 1
            
        except AssertionError as e:
            print(f"  ❌ Failed: {e}")
            results["failed"] += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["failed"] += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Results: {results['passed']} passed, {results['failed']} failed")
    
    return results["failed"] == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)