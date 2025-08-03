import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# Complete Streamlit App Code
# Set page configuration
st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .recommendation-card {
        background-color: #82BFFA;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .segment-result {
        background-color: #BAFFBA;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load trained models and data"""
    try:
        model_path = 'models/'
        
        # Load models
        with open(f'{model_path}kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        
        with open(f'{model_path}scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open(f'{model_path}product_similarity.pkl', 'rb') as f:
            product_similarity_df = pickle.load(f)
        
        # Load data
        rfm_data = pd.read_csv(f'{model_path}rfm_data.csv')
        product_mapping = pd.read_csv(f'{model_path}product_mapping.csv')
        
        # Create cluster mapping
        cluster_mapping = dict(zip(rfm_data['Cluster'], rfm_data['Cluster_Label']))
        
        return kmeans_model, scaler, product_similarity_df, rfm_data, product_mapping, cluster_mapping
        
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run the main analysis script first to generate model files.")
        return None, None, None, None, None, None

def get_product_recommendations(product_name, product_mapping, product_similarity_df, n_recommendations=5):
    """Get product recommendations based on product name"""
    
    # Find stock code for the product name
    matching_products = product_mapping[
        product_mapping['Description'].str.contains(product_name, case=False, na=False)
    ]
    
    if matching_products.empty:
        return None, "Product not found. Please try a different product name."
    
    stock_code = matching_products.iloc[0]['StockCode']
    
    if stock_code not in product_similarity_df.index:
        return None, "Product not found in similarity matrix."
    
    # Get similarity scores
    similar_products = product_similarity_df[stock_code].sort_values(ascending=False)
    recommendations = similar_products.iloc[1:n_recommendations+1]
    
    # Get product descriptions
    rec_data = []
    for stock_code_rec in recommendations.index:
        description = product_mapping[product_mapping['StockCode'] == stock_code_rec]['Description'].iloc[0]
        similarity_score = recommendations[stock_code_rec]
        rec_data.append({
            'Product': description,
            'Similarity Score': f"{similarity_score:.3f}"
        })
    
    return pd.DataFrame(rec_data), None

def predict_customer_segment(recency, frequency, monetary, kmeans_model, scaler, cluster_mapping):
    """Predict customer segment for given RFM values"""
    
    # Scale the input features
    scaled_features = scaler.transform([[recency, frequency, monetary]])
    
    # Predict cluster
    cluster = kmeans_model.predict(scaled_features)[0]
    
    # Map cluster to label
    cluster_label = cluster_mapping.get(cluster, 'Unknown')
    
    return cluster, cluster_label

def main():
    # Title and header
    st.title("🛒 Shopper Spectrum")
    st.markdown("### Customer Segmentation and Product Recommendations")
    st.markdown("---")
    
    # Load models
    kmeans_model, scaler, product_similarity_df, rfm_data, product_mapping, cluster_mapping = load_models()
    
    if kmeans_model is None:
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Product Recommendations", "👥 Customer Segmentation", "📊 Analytics Dashboard"])
    
    # Tab 1: Product Recommendations
    with tab1:
        st.header("🎯 Product Recommendation System")
        st.markdown("Enter a product name to get similar product recommendations based on collaborative filtering.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            product_input = st.text_input(
                "Product Name:",
                placeholder="e.g., WHITE METAL LANTERN, CERAMIC BOWL, etc.",
                help="Enter any part of the product name"
            )
        
        with col2:
            n_recs = st.selectbox("Number of Recommendations:", [3, 5, 7, 10], index=1)
        
        if st.button("🔍 Get Recommendations", type="primary"):
            if product_input:
                with st.spinner("Finding similar products..."):
                    recommendations, error = get_product_recommendations(
                        product_input, product_mapping, product_similarity_df, n_recs
                    )
                
                if error:
                    st.error(error)
                else:
                    st.success(f"Found {len(recommendations)} similar products!")
                    
                    for idx, row in recommendations.iterrows():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>#{idx+1} {row['Product']}</h4>
                            <p><strong>Similarity Score:</strong> {row['Similarity Score']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a product name.")
    
    # Tab 2: Customer Segmentation
    with tab2:
        st.header("👥 Customer Segmentation Predictor")
        st.markdown("Enter customer RFM values to predict their segment.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recency = st.number_input(
                "Recency (Days since last purchase):",
                min_value=0,
                max_value=500,
                value=30,
                help="Number of days since the customer's last purchase"
            )
        
        with col2:
            frequency = st.number_input(
                "Frequency (Number of purchases):",
                min_value=1,
                max_value=100,
                value=5,
                help="Total number of purchases made by the customer"
            )
        
        with col3:
            monetary = st.number_input(
                "Monetary (Total spend):",
                min_value=0.0,
                max_value=10000.0,
                value=500.0,
                step=10.0,
                help="Total amount spent by the customer"
            )
        
        if st.button("🎯 Predict Segment", type="primary"):
            cluster, cluster_label = predict_customer_segment(
                recency, frequency, monetary, kmeans_model, scaler, cluster_mapping
            )
            
            # Define segment descriptions
            segment_descriptions = {
                'High-Value': '🌟 Premium customers who purchase frequently with high spend',
                'Regular': '👍 Steady customers with consistent purchase behavior',
                'Occasional': '📅 Customers who make infrequent purchases',
                'At-Risk': '⚠️ Customers who haven\'t purchased recently and may churn'
            }
            
            description = segment_descriptions.get(cluster_label, 'Unknown segment')
            
            st.markdown(f"""
            <div class="segment-result">
                <h2>Customer Segment: {cluster_label}</h2>
                <p style="font-size: 1.2em;">{description}</p>
                <p><strong>Cluster ID:</strong> {cluster}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show segment characteristics
            if cluster_label in ['High-Value', 'Regular', 'Occasional', 'At-Risk']:
                st.subheader("📈 Segment Characteristics")
                segment_data = rfm_data[rfm_data['Cluster_Label'] == cluster_label]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Recency", f"{segment_data['Recency'].mean():.1f} days")
                with col2:
                    st.metric("Avg Frequency", f"{segment_data['Frequency'].mean():.1f}")
                with col3:
                    st.metric("Avg Monetary", f"${segment_data['Monetary'].mean():.2f}")
                with col4:
                    st.metric("Customers", f"{len(segment_data)}")
    
    # Tab 3: Analytics Dashboard
    with tab3:
        st.header("📊 Analytics Dashboard")
        
        # Segment distribution
        st.subheader("Customer Segment Distribution")
        segment_counts = rfm_data['Cluster_Label'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(segment_counts)
        
        with col2:
            st.write("**Segment Counts:**")
            for segment, count in segment_counts.items():
                percentage = (count / len(rfm_data)) * 100
                st.write(f"• {segment}: {count} ({percentage:.1f}%)")
        
        # RFM Statistics
        st.subheader("RFM Statistics by Segment")
        segment_stats = rfm_data.groupby('Cluster_Label')[['Recency', 'Frequency', 'Monetary']].mean()
        st.dataframe(segment_stats.round(2), use_container_width=True)
        
        # Sample customers from each segment
        st.subheader("Sample Customers by Segment")
        selected_segment = st.selectbox("Select Segment:", rfm_data['Cluster_Label'].unique())
        sample_customers = rfm_data[rfm_data['Cluster_Label'] == selected_segment].head(10)
        st.dataframe(sample_customers[['CustomerID', 'Recency', 'Frequency', 'Monetary']], use_container_width=True)

if __name__ == "__main__":
    main()