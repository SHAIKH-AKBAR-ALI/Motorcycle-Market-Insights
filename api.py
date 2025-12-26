from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="Motorcycle Market API",
    description="API for motorcycle price prediction and market analysis",
    version="1.0.0"
)

# Pydantic models for request/response
class MotorcycleInput(BaseModel):
    model_year: int
    kms_driven: float
    mileage: float
    power: float
    brand: str
    owner: str
    location: str

class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: List[float]
    model_version: str

class MarketStats(BaseModel):
    average_price: float
    median_price: float
    price_range: List[float]
    total_listings: int

# Mock model loading (replace with actual model)
class MockModel:
    def predict(self, X):
        # Simple mock prediction based on features
        return X[:, 3] * 3000 + X[:, 2] * -500 + (2025 - X[:, 0]) * -2000 + np.random.randint(20000, 80000, len(X))

model = MockModel()

@app.get("/")
async def root():
    return {"message": "Motorcycle Market API", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(motorcycle: MotorcycleInput):
    try:
        # Prepare features (simplified)
        features = np.array([[
            motorcycle.model_year,
            motorcycle.kms_driven,
            motorcycle.mileage,
            motorcycle.power,
            1 if motorcycle.brand.lower() == "bajaj" else 0,  # Simplified encoding
            1 if motorcycle.owner == "first owner" else 0
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction = max(prediction, 15000)  # Minimum price
        
        # Calculate confidence interval (mock)
        confidence = [prediction * 0.9, prediction * 1.1]
        
        return PredictionResponse(
            predicted_price=float(prediction),
            confidence_interval=confidence,
            model_version="1.0.0"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/market-stats", response_model=MarketStats)
async def get_market_stats(brand: Optional[str] = None):
    # Mock market statistics
    base_price = 75000
    if brand:
        brand_multiplier = {
            "bajaj": 0.8,
            "royal enfield": 1.2,
            "honda": 1.0,
            "yamaha": 1.1,
            "ktm": 1.3,
            "tvs": 0.7
        }
        multiplier = brand_multiplier.get(brand.lower(), 1.0)
        base_price *= multiplier
    
    return MarketStats(
        average_price=base_price,
        median_price=base_price * 0.9,
        price_range=[base_price * 0.5, base_price * 2.0],
        total_listings=np.random.randint(500, 2000)
    )

@app.get("/brands")
async def get_brands():
    return {
        "brands": ["Bajaj", "Royal Enfield", "Honda", "Yamaha", "KTM", "TVS", "Suzuki", "Kawasaki"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)