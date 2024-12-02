import streamlit as st
import pickle
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Load Random Forest model
with open("random_forest_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Class labels
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize image to 128x128
    image_array = np.array(image).flatten()  # Flatten the image
    scaler = StandardScaler()  # Scale the input if required
    image_array = scaler.fit_transform([image_array])
    return image_array

# Prediction function
def model_prediction(test_image):
    image = Image.open(test_image)  # Open the uploaded image
    processed_image = preprocess_image(image)  # Preprocess the image
    prediction = model.predict(processed_image)  # Predict using Random Forest
    return int(prediction[0])  # Return the class index

# Sidebar
st.sidebar.title("KRISHISUVIDHA")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display the banner image
img = Image.open("Diseases.png")
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        st.image(test_image, width=4, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            st.success(f"Model is Predicting it's a {class_name[result_index]}")
