import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Motorcycle Market Insights",
    page_icon="🏍️",
    layout="wide"
)

# Title and description
st.title("🏍️ Motorcycle Market Insights")
st.markdown("### Predict motorcycle prices and explore market trends")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Price Prediction", "Market Analysis", "Data Explorer"])

if page == "Price Prediction":
    st.header("Motorcycle Price Prediction")
    
    # Input form
    col1, col2 = st.columns(2)
    
    with col1:
        model_year = st.slider("Model Year", 2000, 2025, 2020)
        kms_driven = st.number_input("Kilometers Driven", 0, 200000, 15000)
        mileage = st.slider("Mileage (kmpl)", 10, 100, 40)
    
    with col2:
        power = st.slider("Power (bhp)", 5, 150, 25)
        brand = st.selectbox("Brand", ["Bajaj", "Royal Enfield", "Honda", "Yamaha", "KTM", "TVS"])
        owner = st.selectbox("Owner Type", ["first owner", "second owner", "third owner"])
    
    # Calculate age
    age = 2025 - model_year
    
    # Prediction button
    if st.button("Predict Price"):
        # Mock prediction (replace with actual model)
        predicted_price = (power * 3000) + (mileage * -500) + (age * -2000) + np.random.randint(20000, 80000)
        predicted_price = max(predicted_price, 15000)  # Minimum price
        
        st.success(f"Estimated Price: ₹{predicted_price:,.0f}")
        
        # Price breakdown
        st.subheader("Price Factors")
        factors = {
            "Power": power * 3000,
            "Mileage": mileage * -500,
            "Age": age * -2000,
            "Base Price": 50000
        }
        
        fig = px.bar(x=list(factors.keys()), y=list(factors.values()), 
                    title="Price Factor Breakdown")
        st.plotly_chart(fig)

elif page == "Market Analysis":
    st.header("Market Analysis Dashboard")
    
    # Generate sample data for visualization
    brands = ["Bajaj", "Royal Enfield", "Honda", "Yamaha", "KTM", "TVS", "Suzuki"]
    sample_data = pd.DataFrame({
        'Brand': np.random.choice(brands, 1000),
        'Price': np.random.normal(80000, 30000, 1000),
        'Power': np.random.normal(25, 10, 1000),
        'Mileage': np.random.normal(40, 15, 1000),
        'Age': np.random.randint(1, 15, 1000)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution by brand
        fig1 = px.box(sample_data, x='Brand', y='Price', 
                     title="Price Distribution by Brand")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Power vs Price scatter
        fig2 = px.scatter(sample_data, x='Power', y='Price', color='Brand',
                         title="Power vs Price Analysis")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Market trends
    st.subheader("Market Trends")
    trend_data = pd.DataFrame({
        'Year': range(2015, 2025),
        'Average_Price': [60000, 62000, 65000, 68000, 72000, 75000, 78000, 82000, 85000, 88000]
    })
    
    fig3 = px.line(trend_data, x='Year', y='Average_Price', 
                  title="Average Motorcycle Price Trend")
    st.plotly_chart(fig3, use_container_width=True)

else:  # Data Explorer
    st.header("Data Explorer")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your motorcycle dataset", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Dataset Overview")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head())
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Column selection for visualization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("Select X-axis", numeric_cols)
            with col2:
                y_axis = st.selectbox("Select Y-axis", numeric_cols)
            
            if x_axis and y_axis:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please upload a CSV file to explore the data")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Motorcycle Market Analysis")