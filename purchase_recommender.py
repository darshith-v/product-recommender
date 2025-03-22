import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Smart Purchase Predictor",
    page_icon="🛒",
    layout="wide"
)

# App title and description
st.title("🛒 Smart Purchase Predictor")
st.markdown("""
This app predicts what you might want to purchase based on:
- Your past purchase history
- Seasonal trends
- Contextual data (location, weather, etc.)
""")

# Define categories globally
categories = ['Electronics', 'Clothing', 'Groceries', 'Home Goods', 'Sports Equipment']

# Sidebar for user inputs and navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Your Recommendations", "Purchase History"])

# Load sample data (in a real app, this would be connected to a database)
@st.cache_data
def load_sample_data():
    # Create synthetic user purchase history
    np.random.seed(42)
    n_samples = 1000
    
    # Generate dates across different seasons
    dates = pd.date_range(start='2023-01-01', end='2024-03-01', periods=n_samples)
    
    # Generate temperatures to simulate seasonal weather
    temperatures = np.sin(np.linspace(0, 2*np.pi, n_samples)) * 15 + 20
    
    # Sample data
    data = {
        'date': dates,
        'user_id': np.random.randint(1, 51, n_samples),
        'product_category': np.random.choice(categories, n_samples),
        'amount': np.random.normal(50, 25, n_samples),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
        'weather': temperatures,
        'day_of_week': [d.weekday() for d in dates],
        'month': [d.month for d in dates],
        'is_holiday': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Add some seasonal patterns
    # More clothing purchases in winter, more sports in summer
    winter_months = [12, 1, 2]
    summer_months = [6, 7, 8]
    
    # Adjust product categories based on season
    for i, row in df.iterrows():
        if row['month'] in winter_months and np.random.random() < 0.6:
            df.at[i, 'product_category'] = np.random.choice(['Clothing', 'Home Goods'], p=[0.7, 0.3])
        elif row['month'] in summer_months and np.random.random() < 0.6:
            df.at[i, 'product_category'] = np.random.choice(['Sports Equipment', 'Electronics'], p=[0.6, 0.4])
    
    return df

df = load_sample_data()

# Function to create user profiles/segments
def create_user_segments(df):
    # Get unique product categories from the dataframe
    unique_categories = df['product_category'].unique().tolist()
    
    # Aggregate data by user
    user_profiles = df.groupby('user_id').agg({
        'amount': ['mean', 'sum', 'count'],
        'product_category': lambda x: x.value_counts().index[0],  # most purchased category
        'location': lambda x: x.value_counts().index[0],  # most common location
    }).reset_index()
    
    # Fix the column names - convert MultiIndex to flat string columns
    user_profiles.columns = ['user_id', 'avg_purchase', 'total_spent', 'purchase_count', 'favorite_category', 'primary_location']
    
    # Create category distributions
    category_dummies = pd.get_dummies(df['product_category'])
    category_dist = df.join(category_dummies).groupby('user_id')[unique_categories].mean()
    
    # Join with user profiles
    user_profiles = user_profiles.join(category_dist, on='user_id')
    
    # Create user segments using K-means clustering
    features = ['avg_purchase', 'total_spent', 'purchase_count'] + unique_categories
    X = user_profiles[features].copy()
    
    # Convert ALL column names to strings to prevent type mismatch
    # Convert column names to strings and remove leading/trailing spaces
    X.columns = X.columns.map(lambda x: str(x).strip())

    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    user_profiles['segment'] = kmeans.fit_predict(X_scaled)
    
    # Map segments to descriptive names based on characteristics
    segment_names = {
        0: "Budget Shoppers",
        1: "Tech Enthusiasts",
        2: "Premium Buyers",
        3: "Occasional Shoppers"
    }
    
    user_profiles['segment_name'] = user_profiles['segment'].map(segment_names)
    
    return user_profiles

# Train a simple recommendation model
def train_recommendation_model(df):
    # Feature engineering
    # One-hot encode categorical variables
    df_model = pd.get_dummies(df, columns=['location', 'product_category'], drop_first=True)
    
    # Create season feature
    df_model['season'] = df_model['month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall'
    )
    df_model = pd.get_dummies(df_model, columns=['season'], drop_first=True)
    
    # Select features and target
    features = [col for col in df_model.columns if col not in ['date', 'product_category']]
    
    # Create multiple binary classifiers (one for each product category)
    models = {}
    
    for category in df['product_category'].unique():
        # Create binary target: 1 if purchased this category, 0 otherwise
        y = (df['product_category'] == category).astype(int)
        
        # Ensure all feature names are strings
        X = df_model[features].copy()
        X.columns = X.columns.astype(str)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Store model
        models[category] = model
    
    # Ensure feature list has string column names
    feature_list = [str(col) for col in features]
    
    return models, feature_list

# Dashboard page
if page == "Dashboard":
    st.header("Purchase Trends Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Purchase Trends")
        monthly_data = df.groupby(df['date'].dt.strftime('%Y-%m')).agg({
            'amount': 'sum',
            'user_id': 'nunique'
        }).reset_index()
        monthly_data.columns = ['Month', 'Total Sales', 'Unique Customers']
        
        fig = px.line(monthly_data, x='Month', y=['Total Sales', 'Unique Customers'], 
                     title='Monthly Sales and Customer Trends')
        st.plotly_chart(fig)
    
    with col2:
        st.subheader("Category Distribution")
        category_counts = df['product_category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig = px.pie(category_counts, values='Count', names='Category', 
                    title='Purchase Distribution by Category')
        st.plotly_chart(fig)
    
    # Seasonal trends
    st.subheader("Seasonal Product Popularity")
    seasonal_data = df.groupby(['month', 'product_category']).size().reset_index(name='count')
    fig = px.line(seasonal_data, x='month', y='count', color='product_category',
                 title='Product Category Popularity by Month')
    st.plotly_chart(fig)
    
    # Weather impact
    st.subheader("Weather Impact on Purchases")
    # Create weather bins
    df['temp_range'] = pd.cut(df['weather'], bins=5, labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot'])
    weather_impact = df.groupby(['temp_range', 'product_category']).size().reset_index(name='count')
    
    fig = px.bar(weather_impact, x='temp_range', y='count', color='product_category',
                barmode='group', title='Purchase Patterns by Weather Condition')
    st.plotly_chart(fig)

# Recommendations page
elif page == "Your Recommendations":
    st.header("Your Personalized Recommendations")
    
    # Select a user for demo purposes
    user_id = st.sidebar.selectbox("Select User ID (for demo)", sorted(df['user_id'].unique()))
    
    # Get current contextual information
    st.sidebar.subheader("Current Context")
    current_date = st.sidebar.date_input("Date", datetime.now())
    current_location = st.sidebar.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
    current_weather = st.sidebar.slider("Temperature (°C)", 0, 40, 25)
    
    # Convert date components
    current_month = current_date.month
    current_day = current_date.weekday()
    is_holiday = st.sidebar.checkbox("Is today a holiday?", False)
    
    # Create user segments
    user_profiles = create_user_segments(df)
    
    # Display user profile
    user_profile = user_profiles[user_profiles['user_id'] == user_id].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("User Segment", user_profile['segment_name'])
    with col2:
        st.metric("Favorite Category", user_profile['favorite_category'])
    with col3:
        st.metric("Total Purchases", f"${user_profile['total_spent']:.2f}")
    
    # Train recommendation models
    models, feature_list = train_recommendation_model(df)
    
    # Prepare input for prediction
    user_data = df[df['user_id'] == user_id].iloc[-1:].copy()
    user_data['date'] = pd.to_datetime(current_date)
    user_data['location'] = current_location
    user_data['weather'] = current_weather
    user_data['day_of_week'] = current_day
    user_data['month'] = current_month
    user_data['is_holiday'] = int(is_holiday)
    
    # Transform user data for prediction
    user_data_model = pd.get_dummies(user_data, columns=['location', 'product_category'], drop_first=True)
    user_data_model['season'] = user_data_model['month'].apply(lambda x: 
        'Winter' if x in [12, 1, 2] else
        'Spring' if x in [3, 4, 5] else
        'Summer' if x in [6, 7, 8] else 'Fall'
    )
    user_data_model = pd.get_dummies(user_data_model, columns=['season'], drop_first=True)
    
    # Convert ALL column names to strings
    user_data_model.columns = user_data_model.columns.astype(str)
    
    # Make predictions for each category
    st.subheader("Recommended Products")
    
    recommendations = []
    for category, model in models.items():
        # Ensure all features are present
        input_data = user_data_model.copy()
        
        # Create columns for any missing features
        for feature in feature_list:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Select only the required features and ensure they're in the right order
        input_features = input_data[feature_list]
                
        # Get probability of recommending this category
        try:
            prob = model.predict_proba(input_features)[0][1]
            recommendations.append((category, prob))
        except Exception as e:
            st.error(f"Error predicting for {category}: {e}")
            continue
    
    # Sort by probability
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Display top recommendations
    for i, (category, prob) in enumerate(recommendations[:3]):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(f"https://via.placeholder.com/100?text={category.replace(' ', '+')}", width=100)
        with col2:
            st.subheader(f"{i+1}. {category}")
            st.progress(prob)
            st.write(f"Recommendation confidence: {prob*100:.1f}%")
            
            # Generate dynamic recommendation reason
            if user_profile['favorite_category'] == category:
                reason = "Based on your purchase history"
            elif current_month in [12, 1, 2] and category == "Clothing":
                reason = "Winter season - perfect time for new clothing items"
            elif current_month in [6, 7, 8] and category == "Sports Equipment":
                reason = "Summer season - great time for outdoor activities"
            elif is_holiday:
                reason = "Holiday special recommendation"
            elif category == "Electronics" and current_weather > 30:
                reason = "Stay cool with new electronics during hot weather"
            elif category == "Home Goods" and current_day >= 5:  # Weekend
                reason = "Weekend home improvement recommendation"
            else:
                reason = "Based on users similar to you"
                
            st.write(f"Why: {reason}")

# Purchase History page
else:
    st.header("Your Purchase History")
    
    # Select a user for demo purposes
    user_id = st.sidebar.selectbox("Select User ID (for demo)", sorted(df['user_id'].unique()))
    
    # Filter data for selected user
    user_data = df[df['user_id'] == user_id].sort_values('date', ascending=False)
    
    # Display purchase history
    st.dataframe(
        user_data[['date', 'product_category', 'amount', 'location']].reset_index(drop=True),
        use_container_width=True
    )
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Spent", f"${user_data['amount'].sum():.2f}")
    with col2:
        st.metric("Average Purchase", f"${user_data['amount'].mean():.2f}")
    with col3:
        st.metric("Total Purchases", f"{len(user_data)}")
    
    # Purchase trends over time
    st.subheader("Your Purchase Trends")
    
    monthly_user_data = user_data.groupby(user_data['date'].dt.strftime('%Y-%m')).agg({
        'amount': 'sum'
    }).reset_index()
    monthly_user_data.columns = ['Month', 'Total Spent']
    
    fig = px.line(monthly_user_data, x='Month', y='Total Spent', 
                 title='Your Monthly Spending')
    st.plotly_chart(fig)
    
    # Category breakdown
    st.subheader("Your Category Preferences")
    
    category_user_data = user_data['product_category'].value_counts().reset_index()
    category_user_data.columns = ['Category', 'Count']
    
    fig = px.pie(category_user_data, values='Count', names='Category', 
                title='Your Purchase Distribution by Category')
    st.plotly_chart(fig)