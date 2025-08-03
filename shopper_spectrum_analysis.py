# Shopper Spectrum: Customer Segmentation and Product Recommendations
# Complete Implementation with Product Name Support

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class ShopperSpectrum:
    def __init__(self):
        self.df = None
        self.rfm_df = None
        self.scaled_rfm = None
        self.kmeans_model = None
        self.scaler = None
        self.product_similarity_matrix = None
        self.customer_product_matrix = None
        self.product_mapping = None  # New: Product code to name mapping
        
    def load_and_explore_data(self, file_path):
        """Load and explore the dataset"""
        print("📊 Loading and Exploring Dataset...")
        
        # Load data
        self.df = pd.read_csv(file_path, encoding='latin1')
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nColumn Names: {list(self.df.columns)}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        
        return self.df
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("\n🧹 Data Preprocessing...")
        
        initial_shape = self.df.shape
        print(f"Initial dataset shape: {initial_shape}")
        
        # Remove rows with missing CustomerID
        self.df = self.df.dropna(subset=['CustomerID'])
        print(f"After removing missing CustomerID: {self.df.shape}")
        
        # Convert CustomerID to integer
        self.df['CustomerID'] = self.df['CustomerID'].astype(int)
        
        # Convert InvoiceDate to datetime
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        
        # Remove cancelled invoices (starting with 'C')
        self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
        print(f"After removing cancelled invoices: {self.df.shape}")
        
        # Remove negative or zero quantities and prices
        self.df = self.df[(self.df['Quantity'] > 0) & (self.df['UnitPrice'] > 0)]
        print(f"After removing negative/zero quantities and prices: {self.df.shape}")
        
        # Calculate total amount for each transaction
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        print(f"Final dataset shape after preprocessing: {self.df.shape}")
        
        # Create product mapping (StockCode to Description)
        self.product_mapping = self.df[['StockCode', 'Description']].drop_duplicates()
        self.product_mapping = self.product_mapping.set_index('StockCode')['Description'].to_dict()
        print(f"Created product mapping for {len(self.product_mapping)} unique products")
        
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        print("\n📈 Performing Exploratory Data Analysis...")
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Transaction volume by country (top 10)
        country_counts = self.df['Country'].value_counts().head(10)
        axes[0, 0].bar(range(len(country_counts)), country_counts.values)
        axes[0, 0].set_title('Top 10 Countries by Transaction Volume')
        axes[0, 0].set_xlabel('Country')
        axes[0, 0].set_ylabel('Number of Transactions')
        axes[0, 0].set_xticks(range(len(country_counts)))
        axes[0, 0].set_xticklabels(country_counts.index, rotation=45)
        
        # 2. Top-selling products
        product_sales = self.df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        axes[0, 1].barh(range(len(product_sales)), product_sales.values)
        axes[0, 1].set_title('Top 10 Best-Selling Products')
        axes[0, 1].set_xlabel('Total Quantity Sold')
        axes[0, 1].set_yticks(range(len(product_sales)))
        axes[0, 1].set_yticklabels([desc[:30] + '...' if len(desc) > 30 else desc for desc in product_sales.index])
        
        # 3. Purchase trends over time (monthly)
        self.df['YearMonth'] = self.df['InvoiceDate'].dt.to_period('M')
        monthly_sales = self.df.groupby('YearMonth')['TotalAmount'].sum()
        axes[1, 0].plot(range(len(monthly_sales)), monthly_sales.values, marker='o')
        axes[1, 0].set_title('Monthly Sales Trend')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Total Sales Amount')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Monetary distribution per transaction
        axes[1, 1].hist(self.df['TotalAmount'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Distribution of Transaction Amounts')
        axes[1, 1].set_xlabel('Transaction Amount')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xlim(0, self.df['TotalAmount'].quantile(0.95))  # Remove outliers for better visualization
        
        # 5. Customer transaction frequency
        customer_freq = self.df.groupby('CustomerID').size()
        axes[2, 0].hist(customer_freq, bins=30, alpha=0.7, edgecolor='black')
        axes[2, 0].set_title('Distribution of Customer Transaction Frequency')
        axes[2, 0].set_xlabel('Number of Transactions per Customer')
        axes[2, 0].set_ylabel('Number of Customers')
        
        # 6. Customer spending distribution
        customer_spending = self.df.groupby('CustomerID')['TotalAmount'].sum()
        axes[2, 1].hist(customer_spending, bins=30, alpha=0.7, edgecolor='black')
        axes[2, 1].set_title('Distribution of Customer Total Spending')
        axes[2, 1].set_xlabel('Total Spending per Customer')
        axes[2, 1].set_ylabel('Number of Customers')
        axes[2, 1].set_xlim(0, customer_spending.quantile(0.95))
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n📊 Dataset Summary:")
        print(f"Total Transactions: {len(self.df):,}")
        print(f"Unique Customers: {self.df['CustomerID'].nunique():,}")
        print(f"Unique Products: {self.df['StockCode'].nunique():,}")
        print(f"Date Range: {self.df['InvoiceDate'].min()} to {self.df['InvoiceDate'].max()}")
        print(f"Total Revenue: ${self.df['TotalAmount'].sum():,.2f}")
        print(f"Average Transaction Value: ${self.df['TotalAmount'].mean():.2f}")
        
    def calculate_rfm_features(self):
        """Calculate RFM (Recency, Frequency, Monetary) features"""
        print("\n🔢 Calculating RFM Features...")
        
        # Get the latest date in the dataset
        latest_date = self.df['InvoiceDate'].max()
        print(f"Latest date in dataset: {latest_date}")
        
        # Calculate RFM metrics for each customer
        rfm_data = []
        
        for customer_id in self.df['CustomerID'].unique():
            customer_data = self.df[self.df['CustomerID'] == customer_id]
            
            # Recency: Days since last purchase
            last_purchase_date = customer_data['InvoiceDate'].max()
            recency = (latest_date - last_purchase_date).days
            
            # Frequency: Number of transactions
            frequency = len(customer_data['InvoiceNo'].unique())
            
            # Monetary: Total amount spent
            monetary = customer_data['TotalAmount'].sum()
            
            rfm_data.append({
                'CustomerID': customer_id,
                'Recency': recency,
                'Frequency': frequency,
                'Monetary': monetary
            })
        
        self.rfm_df = pd.DataFrame(rfm_data)
        
        print(f"RFM DataFrame shape: {self.rfm_df.shape}")
        print(f"\nRFM Summary Statistics:")
        print(self.rfm_df.describe())
        
        # Visualize RFM distributions
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('RFM Feature Distributions', fontsize=16, fontweight='bold')
        
        # Recency distribution
        axes[0].hist(self.rfm_df['Recency'], bins=30, alpha=0.7, edgecolor='black')
        axes[0].set_title('Recency Distribution')
        axes[0].set_xlabel('Days Since Last Purchase')
        axes[0].set_ylabel('Number of Customers')
        
        # Frequency distribution
        axes[1].hist(self.rfm_df['Frequency'], bins=30, alpha=0.7, edgecolor='black')
        axes[1].set_title('Frequency Distribution')
        axes[1].set_xlabel('Number of Transactions')
        axes[1].set_ylabel('Number of Customers')
        
        # Monetary distribution
        axes[2].hist(self.rfm_df['Monetary'], bins=30, alpha=0.7, edgecolor='black')
        axes[2].set_title('Monetary Distribution')
        axes[2].set_xlabel('Total Spending')
        axes[2].set_ylabel('Number of Customers')
        
        plt.tight_layout()
        plt.show()
        
        return self.rfm_df
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        print("\n🎯 Finding Optimal Number of Clusters...")
        
        # Standardize RFM features
        self.scaler = StandardScaler()
        self.scaled_rfm = self.scaler.fit_transform(self.rfm_df[['Recency', 'Frequency', 'Monetary']])
        
        # Calculate inertia and silhouette scores for different cluster numbers
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_rfm)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_rfm, kmeans.labels_))
        
        # Plot elbow curve and silhouette scores
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Optimal Cluster Selection', fontsize=16, fontweight='bold')
        
        # Elbow curve
        axes[0].plot(K_range, inertias, 'bo-')
        axes[0].set_title('Elbow Method')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].grid(True)
        
        # Silhouette scores
        axes[1].plot(K_range, silhouette_scores, 'ro-')
        axes[1].set_title('Silhouette Score')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Find optimal k based on highest silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")
        
        return optimal_k
    
    def perform_clustering(self, n_clusters=4):
        """Perform K-means clustering on RFM features"""
        print(f"\n🎯 Performing K-means Clustering with {n_clusters} clusters...")
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(self.scaled_rfm)
        
        # Add cluster labels to RFM dataframe
        self.rfm_df['Cluster'] = clusters
        
        # Calculate cluster centers in original scale
        cluster_centers = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
        cluster_summary = pd.DataFrame(cluster_centers, columns=['Recency', 'Frequency', 'Monetary'])
        cluster_summary['Cluster'] = range(n_clusters)
        cluster_summary['Count'] = self.rfm_df['Cluster'].value_counts().sort_index().values
        
        print("\n📊 Cluster Summary:")
        print(cluster_summary)
        
        # Assign cluster labels based on RFM characteristics
        cluster_labels = {}
        for i in range(n_clusters):
            r = cluster_summary.loc[i, 'Recency']
            f = cluster_summary.loc[i, 'Frequency']
            m = cluster_summary.loc[i, 'Monetary']
            
            if r < self.rfm_df['Recency'].median() and f > self.rfm_df['Frequency'].median() and m > self.rfm_df['Monetary'].median():
                cluster_labels[i] = 'High-Value'
            elif f > self.rfm_df['Frequency'].quantile(0.25) and m > self.rfm_df['Monetary'].quantile(0.25):
                cluster_labels[i] = 'Regular'
            elif r > self.rfm_df['Recency'].median():
                cluster_labels[i] = 'At-Risk'
            else:
                cluster_labels[i] = 'Occasional'
        
        # Add cluster labels
        self.rfm_df['Cluster_Label'] = self.rfm_df['Cluster'].map(cluster_labels)
        cluster_summary['Label'] = cluster_summary['Cluster'].map(cluster_labels)
        
        print("\n🏷️ Cluster Labels:")
        print(cluster_summary[['Cluster', 'Label', 'Count', 'Recency', 'Frequency', 'Monetary']])
        
        # Visualize clusters
        self.visualize_clusters()
        
        return cluster_summary
    
    def visualize_clusters(self):
        """Visualize customer clusters"""
        print("\n📊 Visualizing Customer Clusters...")
        
        # Create 3D scatter plot
        fig = plt.figure(figsize=(15, 5))
        
        # 2D scatter plots
        axes = []
        axes.append(fig.add_subplot(131))
        axes.append(fig.add_subplot(132))
        axes.append(fig.add_subplot(133))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        # Recency vs Frequency
        for cluster in self.rfm_df['Cluster'].unique():
            cluster_data = self.rfm_df[self.rfm_df['Cluster'] == cluster]
            axes[0].scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                           c=colors[cluster], alpha=0.6, label=f'Cluster {cluster}')
        axes[0].set_xlabel('Recency (Days)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Recency vs Frequency')
        axes[0].legend()
        
        # Frequency vs Monetary
        for cluster in self.rfm_df['Cluster'].unique():
            cluster_data = self.rfm_df[self.rfm_df['Cluster'] == cluster]
            axes[1].scatter(cluster_data['Frequency'], cluster_data['Monetary'], 
                           c=colors[cluster], alpha=0.6, label=f'Cluster {cluster}')
        axes[1].set_xlabel('Frequency')
        axes[1].set_ylabel('Monetary')
        axes[1].set_title('Frequency vs Monetary')
        axes[1].legend()
        
        # Recency vs Monetary
        for cluster in self.rfm_df['Cluster'].unique():
            cluster_data = self.rfm_df[self.rfm_df['Cluster'] == cluster]
            axes[2].scatter(cluster_data['Recency'], cluster_data['Monetary'], 
                           c=colors[cluster], alpha=0.6, label=f'Cluster {cluster}')
        axes[2].set_xlabel('Recency (Days)')
        axes[2].set_ylabel('Monetary')
        axes[2].set_title('Recency vs Monetary')
        axes[2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Cluster distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = self.rfm_df['Cluster_Label'].value_counts()
        plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('Customer Segment Distribution')
        plt.axis('equal')
        plt.show()
    
    def build_recommendation_system(self):
        """Build item-based collaborative filtering recommendation system"""
        print("\n🤖 Building Product Recommendation System...")
        
        # Create customer-product matrix
        self.customer_product_matrix = self.df.pivot_table(
            index='CustomerID', 
            columns='StockCode', 
            values='Quantity', 
            fill_value=0
        )
        
        print(f"Customer-Product Matrix Shape: {self.customer_product_matrix.shape}")
        
        # Calculate item-item similarity using cosine similarity
        # Transpose to get product-customer matrix for item-based filtering
        product_customer_matrix = self.customer_product_matrix.T
        
        # Calculate cosine similarity between products
        self.product_similarity_matrix = cosine_similarity(product_customer_matrix)
        
        # Convert to DataFrame for easier handling
        self.product_similarity_df = pd.DataFrame(
            self.product_similarity_matrix,
            index=product_customer_matrix.index,
            columns=product_customer_matrix.index
        )
        
        print("✅ Product recommendation system built successfully!")
        
        return self.product_similarity_df
    
    def search_product_by_name(self, product_name_query):
        """Search for products by name/description"""
        if not hasattr(self, 'product_mapping') or self.product_mapping is None:
            return "Product mapping not available. Please run preprocessing first."
        
        # Search for products containing the query string (case-insensitive)
        matching_products = []
        for stock_code, description in self.product_mapping.items():
            if product_name_query.lower() in description.lower():
                matching_products.append({
                    'StockCode': stock_code,
                    'Description': description
                })
        
        if not matching_products:
            return f"No products found matching '{product_name_query}'"
        
        return pd.DataFrame(matching_products)
    
    def get_product_recommendations_by_name(self, product_name_query, n_recommendations=5):
        """Get product recommendations by searching with product name"""
        # First, search for products matching the name
        search_results = self.search_product_by_name(product_name_query)
        
        if isinstance(search_results, str):  # No products found
            return search_results
        
        if len(search_results) == 0:
            return f"No products found matching '{product_name_query}'"
        
        # If multiple products found, show them and use the first one for recommendations
        if len(search_results) > 1:
            print(f"Found {len(search_results)} products matching '{product_name_query}':")
            print(search_results)
            print(f"\nUsing the first product for recommendations: {search_results.iloc[0]['Description']}")
        
        # Get the stock code of the first matching product
        stock_code = search_results.iloc[0]['StockCode']
        
        # Get recommendations for this stock code
        return self.get_product_recommendations(stock_code, n_recommendations)
    
    def get_product_recommendations(self, stock_code, n_recommendations=5):
        """Get product recommendations for a given stock code"""
        if stock_code not in self.product_similarity_df.index:
            return f"Product {stock_code} not found in the dataset"
        
        # Get similarity scores for the given product
        similar_products = self.product_similarity_df[stock_code].sort_values(ascending=False)
        
        # Exclude the product itself and get top N recommendations
        recommendations = similar_products.iloc[1:n_recommendations+1]
        
        # Get product descriptions
        product_info = []
        for stock_code_rec in recommendations.index:
            description = self.product_mapping.get(stock_code_rec, "Description not available")
            similarity_score = recommendations[stock_code_rec]
            product_info.append({
                'StockCode': stock_code_rec,
                'Description': description,
                'Similarity_Score': similarity_score
            })
        
        return pd.DataFrame(product_info)
    
    def get_all_products(self, limit=None):
        """Get all products in the dataset"""
        if not hasattr(self, 'product_mapping') or self.product_mapping is None:
            return "Product mapping not available. Please run preprocessing first."
        
        products = []
        for stock_code, description in self.product_mapping.items():
            products.append({
                'StockCode': stock_code,
                'Description': description
            })
        
        df_products = pd.DataFrame(products)
        
        if limit:
            return df_products.head(limit)
        
        return df_products
    
    def predict_customer_segment(self, recency, frequency, monetary):
        """Predict customer segment for given RFM values"""
        if self.kmeans_model is None or self.scaler is None:
            return "Model not trained yet. Please run clustering first."
        
        # Scale the input features
        scaled_features = self.scaler.transform([[recency, frequency, monetary]])
        
        # Predict cluster
        cluster = self.kmeans_model.predict(scaled_features)[0]
        
        # Map cluster to label
        cluster_mapping = dict(zip(self.rfm_df['Cluster'], self.rfm_df['Cluster_Label']))
        cluster_label = cluster_mapping[cluster]
        
        return {
            'Cluster': cluster,
            'Cluster_Label': cluster_label,
            'Confidence': 'High'  # You could calculate actual confidence if needed
        }
    
    def save_models(self, model_path='models/'):
        """Save trained models for Streamlit app"""
        import pickle
        import os
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        # Save models
        with open(f'{model_path}kmeans_model.pkl', 'wb') as f:
            pickle.dump(self.kmeans_model, f)
        
        with open(f'{model_path}scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{model_path}product_similarity.pkl', 'wb') as f:
            pickle.dump(self.product_similarity_df, f)
        
        # Save product mapping
        with open(f'{model_path}product_mapping.pkl', 'wb') as f:
            pickle.dump(self.product_mapping, f)
        
        # Save processed data
        self.rfm_df.to_csv(f'{model_path}rfm_data.csv', index=False)
        
        # Save product mapping as CSV for easy viewing
        product_mapping_df = pd.DataFrame(list(self.product_mapping.items()), 
                                        columns=['StockCode', 'Description'])
        product_mapping_df.to_csv(f'{model_path}product_mapping.csv', index=False)
        
        print("✅ Models and data saved successfully!")

def main():
    """Main execution function"""
    print("🛒 SHOPPER SPECTRUM: Customer Segmentation and Product Recommendations")
    print("=" * 70)
    
    # Initialize the class
    shopper = ShopperSpectrum()
    
    # Step 1: Load and explore data
    # Note: You'll need to provide the actual dataset path
    print("📌 Step 1: Dataset Collection and Understanding")
    # df = shopper.load_and_explore_data('your_dataset.csv')
    
    # For demonstration, let's create sample data
    # Replace this with actual data loading
    sample_data = create_sample_data()
    shopper.df = sample_data
    
    # Step 2: Data preprocessing
    print("\n📌 Step 2: Data Preprocessing")
    shopper.preprocess_data()
    
    # Step 3: Exploratory Data Analysis
    print("\n📌 Step 3: Exploratory Data Analysis")
    shopper.exploratory_data_analysis()
    
    # Step 4: RFM Feature Engineering
    print("\n📌 Step 4: RFM Feature Engineering")
    shopper.calculate_rfm_features()
    
    # Step 5: Find optimal clusters
    print("\n📌 Step 5: Finding Optimal Clusters")
    optimal_k = shopper.find_optimal_clusters()
    
    # Step 6: Perform clustering
    print("\n📌 Step 6: Customer Segmentation")
    cluster_summary = shopper.perform_clustering(n_clusters=optimal_k)
    
    # Step 7: Build recommendation system
    print("\n📌 Step 7: Building Recommendation System")
    shopper.build_recommendation_system()
    
    # Step 8: Save models for Streamlit
    print("\n📌 Step 8: Saving Models")
    shopper.save_models()
    
    # Example usage
    print("\n📌 Example Usage:")
    
    # Example: Search products by name
    print("\n🔍 Product Search Examples:")
    search_results = shopper.search_product_by_name("mug")
    if isinstance(search_results, pd.DataFrame):
        print("Products containing 'mug':")
        print(search_results.head())
    
    # Example: Get recommendations by product name
    print("\n🎯 Product Recommendations by Name:")
    recommendations = shopper.get_product_recommendations_by_name("mug", n_recommendations=3)
    if isinstance(recommendations, pd.DataFrame):
        print("Recommendations for products containing 'mug':")
        print(recommendations)
    
    # Example recommendation using stock code (traditional method)
    if len(shopper.df) > 0:
        sample_stock_code = shopper.df['StockCode'].iloc[0]
        recommendations = shopper.get_product_recommendations(sample_stock_code)
        print(f"\nRecommendations for product {sample_stock_code}:")
        print(recommendations)
    
    # Example customer segmentation prediction
    prediction = shopper.predict_customer_segment(recency=30, frequency=5, monetary=500)
    print(f"\nCustomer segment prediction for R=30, F=5, M=500:")
    print(prediction)
    
    # Show available products
    print("\n📋 Sample Products Available:")
    all_products = shopper.get_all_products(limit=10)
    if isinstance(all_products, pd.DataFrame):
        print(all_products)
    
    print("\n✅ Project completed successfully!")
    print("Next step: Create Streamlit app using the saved models")

def create_sample_data():
    """Create sample e-commerce data for demonstration"""
    np.random.seed(42)
    
    # Generate sample data with realistic product names
    n_transactions = 10000
    n_customers = 2000
    
    # Sample product categories and names
    product_names = [
        "White Ceramic Mug", "Black Coffee Mug", "Travel Mug Stainless Steel",
        "Ceramic Tea Cup", "Glass Water Bottle", "Insulated Coffee Cup",
        "Vintage Style Mug", "Blue Ceramic Bowl", "Stainless Steel Tumbler",
        "Porcelain Dinner Plate", "Glass Storage Jar", "Bamboo Cutting Board",
        "Wooden Spoon Set", "Silicone Spatula", "Non-stick Frying Pan",
        "Stainless Steel Pot", "Glass Baking Dish", "Ceramic Serving Bowl",
        "Kitchen Timer Digital", "Measuring Cup Set", "Chef Knife Professional",
        "Cutting Board Plastic", "Mixing Bowl Set", "Can Opener Manual",
        "Bottle Opener Metal", "Kitchen Scale Digital", "Toaster 2-Slice",
        "Blender Personal Size", "Food Processor Mini", "Coffee Maker Drip",
        "Electric Kettle", "Microwave Safe Plate", "Oven Mitt Heat Resistant",
        "Kitchen Towel Cotton", "Dish Soap Organic", "Sponge Pack Natural",
        "Cleaning Cloth Microfiber", "Storage Container Glass", "Lunch Box Kids",
        "Thermos Flask", "Water Filter Pitcher", "Ice Cube Tray", "Coaster Set Wood",
        "Placemat Fabric", "Napkin Holder Metal", "Salt Pepper Shaker", "Sugar Bowl White",
        "Milk Jug Glass", "Bread Box Metal", "Fruit Bowl Ceramic", "Wine Glass Set",
        "Beer Mug Large", "Shot Glass Pack", "Champagne Flute", "Cocktail Shaker",
        "Tea Infuser Mesh", "Coffee Grinder Manual", "Espresso Cup Small",
        "Soup Ladle Steel", "Pasta Spoon Wooden", "Whisk Wire Medium",
        "Rolling Pin Wood", "Cookie Cutter Set", "Baking Sheet Non-stick",
        "Muffin Tin 12-Cup", "Loaf Pan Rectangle", "Cake Pan Round",
        "Pie Dish Glass", "Casserole Dish Ceramic", "Roasting Pan Large",
        "Grill Pan Cast Iron", "Wok Carbon Steel", "Sauce Pan Small",
        "Stock Pot Large", "Pressure Cooker Electric", "Slow Cooker 6-Qt",
        "Rice Cooker Digital", "Air Fryer Compact", "Stand Mixer Bowl",
        "Hand Mixer Electric", "Food Storage Bags", "Aluminum Foil Roll",
        "Plastic Wrap Roll", "Parchment Paper", "Freezer Bags Zip",
        "Vacuum Seal Bags", "Food Label Stickers", "Measuring Spoons Set",
        "Funnel Kitchen Small", "Strainer Fine Mesh", "Colander Large",
        "Salad Spinner Plastic", "Vegetable Peeler", "Garlic Press Steel",
        "Lemon Squeezer", "Apple Corer Tool", "Pizza Cutter Wheel",
        "Cheese Grater Box", "Mandoline Slicer", "Kitchen Shears Heavy",
        "Tongs Silicone Tip", "Ladle Soup Large", "Serving Spoon Set",
        "Dinner Fork Set", "Steak Knife Set", "Butter Knife Pack",
        "Soup Spoon Set", "Dessert Spoon Small", "Chopsticks Bamboo",
        "Napkin Ring Set", "Tablecloth Cotton", "Table Runner Linen",
        "Candle Holder Glass", "Centerpiece Bowl", "Serving Tray Wood",
        "Lazy Susan Rotating", "Trivet Heat Resistant", "Pot Holder Fabric",
        "Oven Thermometer", "Meat Thermometer Digital", "Kitchen Scale Analog",
        "Timer Magnetic Back", "Apron Cotton Adult", "Chef Hat White",
        "Dish Rack Stainless", "Soap Dispenser Pump", "Scrub Brush Handle",
        "Dishwasher Tablets", "All-Purpose Cleaner", "Degreaser Kitchen"
    ]
    
    data = []
    
    for i in range(n_transactions):
        invoice_no = f"INV{i+1000:05d}"
        
        # Select random product
        product_idx = np.random.randint(0, len(product_names))
        stock_code = f"PROD{product_idx+1:03d}"
        description = product_names[product_idx]
        
        quantity = np.random.randint(1, 20)
        invoice_date = pd.Timestamp('2022-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
        unit_price = np.random.uniform(5, 100)
        customer_id = np.random.randint(1, n_customers+1)
        country = np.random.choice(['UK', 'Germany', 'France', 'Spain', 'Netherlands'], 
                                  p=[0.7, 0.1, 0.1, 0.05, 0.05])
        
        data.append({
            'InvoiceNo': invoice_no,
            'StockCode': stock_code,
            'Description': description,
            'Quantity': quantity,
            'InvoiceDate': invoice_date,
            'UnitPrice': unit_price,
            'CustomerID': customer_id,
            'Country': country
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
    