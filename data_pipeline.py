import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Any
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotorcycleDataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_columns = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load motorcycle data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data"""
        logger.info("Starting data cleaning...")
        
        # Drop rows with missing values in critical columns
        df = df.dropna(subset=['price', 'model_year', 'power', 'mileage'])
        
        # Clean numerical columns
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        if 'kms_driven' in df.columns:
            df['kms_driven'] = df['kms_driven'].astype(str).str.replace(' Km', '').str.replace('Mileage ', '')
            df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
        
        if 'mileage' in df.columns:
            df['mileage'] = df['mileage'].astype(str).str.replace(' kmpl', '').str.replace('\\n\\n ', '')
            df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
        
        if 'power' in df.columns:
            df['power'] = df['power'].astype(str).str.replace(' bhp', '')
            df['power'] = pd.to_numeric(df['power'], errors='coerce')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle outliers (cap at 99th percentile)
        numerical_cols = ['price', 'kms_driven', 'mileage', 'power']
        for col in numerical_cols:
            if col in df.columns:
                upper_limit = df[col].quantile(0.99)
                df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
        
        logger.info(f"Data cleaned. Final shape: {df.shape}")
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing data"""
        logger.info("Starting feature engineering...")
        
        # Extract brand from model name
        if 'model_name' in df.columns:
            df['brand'] = df['model_name'].apply(lambda x: str(x).split(' ')[0])
        
        # Calculate age
        if 'model_year' in df.columns:
            df['age'] = 2025 - df['model_year']
        
        # Create power-to-weight ratio (if weight data available)
        # df['power_to_weight'] = df['power'] / df['weight']  # Uncomment if weight data available
        
        # Create efficiency score
        if 'mileage' in df.columns and 'power' in df.columns:
            df['efficiency_score'] = df['mileage'] / df['power']
        
        # Price per kilometer (depreciation indicator)
        if 'price' in df.columns and 'kms_driven' in df.columns:
            df['price_per_km'] = df['price'] / (df['kms_driven'] + 1)  # +1 to avoid division by zero
        
        logger.info("Feature engineering completed")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_cols = ['brand', 'owner', 'location']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = df[col].apply(
                            lambda x: self.label_encoders[col].transform([str(x)])[0] 
                            if str(x) in self.label_encoders[col].classes_ else -1
                        )
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'price') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training"""
        # Select feature columns
        feature_cols = ['model_year', 'kms_driven', 'mileage', 'power', 'age']
        
        # Add encoded categorical features
        encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
        feature_cols.extend(encoded_cols)
        
        # Add engineered features
        if 'efficiency_score' in df.columns:
            feature_cols.append('efficiency_score')
        if 'price_per_km' in df.columns and target_col != 'price':
            feature_cols.append('price_per_km')
        
        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else None
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the prediction model"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model trained. Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_pipeline(self, path: str):
        """Save the trained pipeline"""
        pipeline_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        joblib.dump(pipeline_data, path)
        logger.info(f"Pipeline saved to {path}")
    
    def load_pipeline(self, path: str):
        """Load a trained pipeline"""
        pipeline_data = joblib.load(path)
        self.model = pipeline_data['model']
        self.scaler = pipeline_data['scaler']
        self.label_encoders = pipeline_data['label_encoders']
        self.feature_columns = pipeline_data['feature_columns']
        logger.info(f"Pipeline loaded from {path}")

def main():
    """Main pipeline execution"""
    pipeline = MotorcycleDataPipeline()
    
    # Check if data file exists
    data_file = "used_bikes.csv"  # Update with actual file path
    if not os.path.exists(data_file):
        logger.warning(f"Data file {data_file} not found. Creating sample data...")
        # Create sample data for demonstration
        sample_data = pd.DataFrame({
            'model_name': ['Bajaj Pulsar 150'] * 100 + ['Royal Enfield Classic 350'] * 100,
            'model_year': np.random.randint(2010, 2024, 200),
            'kms_driven': np.random.randint(5000, 80000, 200),
            'mileage': np.random.normal(40, 10, 200),
            'power': np.random.normal(20, 5, 200),
            'price': np.random.normal(75000, 25000, 200),
            'owner': np.random.choice(['first owner', 'second owner'], 200),
            'location': np.random.choice(['bangalore', 'mumbai', 'delhi'], 200)
        })
        sample_data.to_csv(data_file, index=False)
    
    try:
        # Load and process data
        df = pipeline.load_data(data_file)
        df = pipeline.clean_data(df)
        df = pipeline.feature_engineering(df)
        df = pipeline.encode_categorical_features(df)
        
        # Prepare features and train model
        X, y = pipeline.prepare_features(df)
        results = pipeline.train_model(X, y)
        
        # Save pipeline
        pipeline.save_pipeline("motorcycle_model.pkl")
        
        print("Pipeline Results:")
        print(f"Train R²: {results['train_score']:.3f}")
        print(f"Test R²: {results['test_score']:.3f}")
        print("\nFeature Importance:")
        for feature, importance in results['feature_importance'].items():
            print(f"{feature}: {importance:.3f}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()