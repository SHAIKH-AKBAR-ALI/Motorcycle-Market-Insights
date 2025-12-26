import pytest
import pandas as pd
import numpy as np
from data_pipeline import MotorcycleDataPipeline

class TestMotorcycleDataPipeline:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'model_name': ['Bajaj Pulsar 150', 'Royal Enfield Classic 350'],
            'model_year': [2020, 2019],
            'kms_driven': ['15000 Km', '25000 Km'],
            'mileage': ['45 kmpl', '35 kmpl'],
            'power': ['14 bhp', '19.8 bhp'],
            'price': [75000, 120000],
            'owner': ['first owner', 'first owner'],
            'location': ['bangalore', 'mumbai']
        })
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return MotorcycleDataPipeline()
    
    def test_clean_data(self, pipeline, sample_data):
        """Test data cleaning functionality"""
        cleaned_data = pipeline.clean_data(sample_data)
        
        # Check that numerical columns are properly converted
        assert cleaned_data['kms_driven'].dtype in [np.float64, np.int64]
        assert cleaned_data['mileage'].dtype in [np.float64, np.int64]
        assert cleaned_data['power'].dtype in [np.float64, np.int64]
        
        # Check no missing values in critical columns
        assert not cleaned_data[['price', 'model_year', 'power', 'mileage']].isnull().any().any()
    
    def test_feature_engineering(self, pipeline, sample_data):
        """Test feature engineering"""
        cleaned_data = pipeline.clean_data(sample_data)
        engineered_data = pipeline.feature_engineering(cleaned_data)
        
        # Check new features are created
        assert 'brand' in engineered_data.columns
        assert 'age' in engineered_data.columns
        
        # Check brand extraction
        assert engineered_data['brand'].iloc[0] == 'Bajaj'
        assert engineered_data['brand'].iloc[1] == 'Royal'
        
        # Check age calculation
        assert engineered_data['age'].iloc[0] == 2025 - 2020
    
    def test_encode_categorical_features(self, pipeline, sample_data):
        """Test categorical encoding"""
        cleaned_data = pipeline.clean_data(sample_data)
        engineered_data = pipeline.feature_engineering(cleaned_data)
        encoded_data = pipeline.encode_categorical_features(engineered_data)
        
        # Check encoded columns are created
        encoded_cols = [col for col in encoded_data.columns if col.endswith('_encoded')]
        assert len(encoded_cols) > 0
    
    def test_prepare_features(self, pipeline, sample_data):
        """Test feature preparation"""
        cleaned_data = pipeline.clean_data(sample_data)
        engineered_data = pipeline.feature_engineering(cleaned_data)
        encoded_data = pipeline.encode_categorical_features(engineered_data)
        
        X, y = pipeline.prepare_features(encoded_data)
        
        # Check shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        assert len(X.columns) > 0
    
    def test_model_training(self, pipeline, sample_data):
        """Test model training with minimal data"""
        # Create larger sample for training
        large_sample = pd.concat([sample_data] * 50, ignore_index=True)
        large_sample['price'] = np.random.normal(80000, 20000, len(large_sample))
        
        cleaned_data = pipeline.clean_data(large_sample)
        engineered_data = pipeline.feature_engineering(cleaned_data)
        encoded_data = pipeline.encode_categorical_features(engineered_data)
        
        X, y = pipeline.prepare_features(encoded_data)
        results = pipeline.train_model(X, y)
        
        # Check model is trained
        assert pipeline.model is not None
        assert 'train_score' in results
        assert 'test_score' in results
        
        # Check predictions work
        predictions = pipeline.predict(X.head(5))
        assert len(predictions) == 5