import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

class TestMotorcycleAPI:
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Motorcycle Market API" in response.json()["message"]
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_get_brands(self):
        """Test brands endpoint"""
        response = client.get("/brands")
        assert response.status_code == 200
        brands = response.json()["brands"]
        assert isinstance(brands, list)
        assert len(brands) > 0
        assert "Bajaj" in brands
    
    def test_predict_price(self):
        """Test price prediction endpoint"""
        test_data = {
            "model_year": 2020,
            "kms_driven": 15000,
            "mileage": 45.0,
            "power": 14.0,
            "brand": "Bajaj",
            "owner": "first owner",
            "location": "bangalore"
        }
        
        response = client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        result = response.json()
        assert "predicted_price" in result
        assert "confidence_interval" in result
        assert "model_version" in result
        assert result["predicted_price"] > 0
    
    def test_market_stats(self):
        """Test market statistics endpoint"""
        response = client.get("/market-stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert "average_price" in stats
        assert "median_price" in stats
        assert "price_range" in stats
        assert "total_listings" in stats
    
    def test_market_stats_with_brand(self):
        """Test market statistics with brand filter"""
        response = client.get("/market-stats?brand=bajaj")
        assert response.status_code == 200
        
        stats = response.json()
        assert stats["average_price"] > 0
    
    def test_invalid_prediction_data(self):
        """Test prediction with invalid data"""
        invalid_data = {
            "model_year": "invalid",
            "kms_driven": -1000,
            "mileage": 45.0,
            "power": 14.0,
            "brand": "Bajaj",
            "owner": "first owner",
            "location": "bangalore"
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error