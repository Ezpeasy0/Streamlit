import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Start on the home page

# Function to navigate to home
def go_home():
    st.session_state.page = 'home'


# Load the trained model and preprocessor
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('kmeans_model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Load the data
data = pd.read_csv('shopping_trends.csv')

# Ensure "Cluster" column exists
if 'Cluster' not in data.columns:
    st.error("Error: 'Cluster' column is missing. Run trained_model.py first.")
    st.stop()


# Calculate cluster metrics
cluster_info = data.groupby('Cluster').agg({
    'Age': 'mean',
    'Purchase Amount (USD)': 'mean',
    'Previous Purchases': 'mean',
    'Gender': lambda x: (x == 'Male').mean(),  # Percentage of Males
    'Customer ID': 'size'  # Cluster size
}).rename(columns={'Age': 'Average Age', 'Gender': 'Percentage Male', 'Customer ID': 'Cluster Size'})

# Streamlit layout for the home page
if st.session_state.page == 'home':
    st.title('Customer Segmentation Based on Shopping Trends')
    st.header("Data Overview")
    st.dataframe(data)

    # Image and caption mappings (Updated)
    image_info = {
        0: [("images/dress0.png", "Dress"), ("images/blouse0.png", "Blouse"), ("images/jewelry0.png", "Jewelry")],
        1: [("images/jewelry1.png", "Jewelry"), ("images/coat1.png", "Coat"), ("images/jacket1.png", "Jacket")],
        2: [("images/belt2.png", "Belt"), ("images/skirt2.png", "Skirt"), ("images/gloves2.png", "Gloves")],
        3: [("images/shirt3.png", "Shirt"), ("images/sunglasses3.png", "Sunglasses"), ("images/pants3.png", "Pants")]
    }

    # Cluster Overview
    st.header("Cluster Overview")
    for i in range(4):
        st.subheader(f"Cluster {i}")
        st.dataframe(cluster_info.loc[i].to_frame().T)  # Display as a table

        # Display 3 images per cluster in columns
        cols = st.columns(3)
        for idx, (img_path, caption) in enumerate(image_info[i]):
            with cols[idx]:
                st.image(img_path, caption=caption, use_container_width=True)

    # Sidebar for Customer Input
    st.sidebar.title("Customer Profile Analysis")
    age_inp = st.sidebar.number_input("Input Age", min_value=0, step=1, format="%d")
    purchase_amount_inp = st.sidebar.number_input("Input Purchase Amount (USD)", min_value=0, step=1, format="%d")  # Changed to integer
    previous_purchase_inp = st.sidebar.number_input("Input Previous Purchases", min_value=0, step=1, format="%d")
    frequency_purchases_inp = st.sidebar.number_input("Input Frequency of Purchases", min_value=0, step=1, format="%d")
    predict_button = st.sidebar.button("Predict")

    # Perform Prediction
    if predict_button:
        input_data = pd.DataFrame([[age_inp, purchase_amount_inp, previous_purchase_inp, frequency_purchases_inp]])
        processed_data = preprocessor.transform(input_data)
        cluster_prediction = kmeans.predict(processed_data)[0]

        # Show predicted cluster result
        st.sidebar.subheader(f"Predicted Cluster: {cluster_prediction}")

        # Navigate to page of cluster prediction
        if cluster_prediction == 0:
            st.session_state.page = 'page1'
            st.rerun()  # Refresh the app 
        elif cluster_prediction == 1:
            st.session_state.page = 'page2'
            st.rerun()  # Refresh the app 
        elif cluster_prediction == 2:
            st.session_state.page = 'page3'
            st.rerun()  # Refresh the app 
        elif cluster_prediction == 3:
            st.session_state.page = 'page4'
            st.rerun()  # Refresh the app 

# Page 1
elif st.session_state.page == 'page1':
    st.title("Cluster0")
    
    # Use st.components.v1.html to render the iframe
    st.components.v1.html(
        """
        <iframe
            src="https://ieny4uefywdpjfyra3zynv.streamlit.app?embed=true"
            style="height: 450px; width: 100%;"
            frameborder="0">
        </iframe>
        """,
        height=450,  
        width=700,   
    )

    # Button to go back to Home
    if st.button("Back to Home"):
        go_home()

# Page 2
elif st.session_state.page == 'page2':
    st.title("Cluster1")
    
    # Use st.components.v1.html to render the iframe
    st.components.v1.html(
        """
        <iframe
            src="https://3ibapx9dzf7okbxscer7dy.streamlit.app?embed=true"
            style="height: 450px; width: 100%;"
            frameborder="0">
        </iframe>
        """,
        height=450,  
        width=700,   
    )
    
    # Button to go back to Home
    if st.button("Back to Home"):
        go_home()

# Page 3
elif st.session_state.page == 'page3':
    st.title("Cluster2")
    
    # Use st.components.v1.html to render the iframe
    st.components.v1.html(
        """
        <iframe
            src="https://kchuujb7snk3yjvnzj77qm.streamlit.app?embed=true"
            style="height: 450px; width: 100%;"
            frameborder="0">
        </iframe>
        """,
        height=450,  
        width=700,   
    )

    # Button to go back to Home
    if st.button("Back to Home"):
        go_home()

# Page 4
elif st.session_state.page == 'page4':
    st.title("Cluster3")
    
    # Use st.components.v1.html to render the iframe
    st.components.v1.html(
        """
        <iframe
            src="https://sdaeqq33tfvf5ziw5e3mqr.streamlit.app?embed=true"
            style="height: 450px; width: 100%;"
            frameborder="0">
        </iframe>
        """,
        height=450,  
        width=700,   
    )

    # Button to go back to Home
    if st.button("Back to Home"):
        go_home()