import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# Load data
data = pd.read_csv('shopping_trends.csv')

# Preprocessing for clustering
numerical_features = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases']
categorical_features = ['Gender', 'Category', 'Item Purchased']
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

data_processed = preprocessor.fit_transform(data)
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_processed)

# Calculate cluster metrics
cluster_info = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Purchase Amount (USD)': 'mean',
    'Previous Purchases': 'mean',
    'Gender': lambda x: (x == 'Male').mean(),  # Percentage Male
    'Customer ID': 'size'  # Cluster size
}).rename(columns={'Age': 'Average Age', 'Gender': 'Percentage Male', 'Customer ID': 'Cluster Size'})

# Streamlit layout
st.title('Customer Segmentation Based on Shopping Trends')
st.header("Data Overview")
st.dataframe(data)

# Image and caption mappings
image_info = {
    0: [("images/dress0.png", "Dress"), ("images/blouse0.png", "Blouse"), ("images/jewelry0.png", "Jewelry") ],
    1: [("images/jewelry1.png", "Jewelry"), ("images/coat1.png", "Coat"), ("images/jacket1.png", "Jacket")],
    2: [("images/belt2.png", "Belt"), ("images/skirt2.png", "Skirt"), ("images/gloves2.png", "Gloves")],
    3: [("images/shirt3.png", "Shirt"), ("images/sunglasses3.png", "Sunglasses"), ("images/pants3.png", "Pants")]
}

st.header("Cluster Overview")
for i in range(4):
    st.subheader(f"Cluster {i}")
    cluster_metrics = cluster_info.loc[i]
    st.write(cluster_metrics)

# Display 3 images per cluster in columns
    cols = st.columns(3)
    for idx, (img_path, caption) in enumerate(image_info[i]):
        with cols[idx]:
            st.image(img_path, caption=caption, use_container_width=True)
   # st.image(image_info[i][0][0], caption=image_info[i][0][1], use_container_width=True)

# Sidebar for customer input
st.sidebar.title("Customer Profile Analysis")
age_inp = st.sidebar.number_input("Input Age")
purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount(USD)")
previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases")
frequency_purchases_inp = st.sidebar.number_input("Input Frequency of Purchases")
submit_button = st.sidebar.button("Submit")