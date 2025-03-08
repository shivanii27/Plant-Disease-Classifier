import streamlit as st
import torch
import json
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plant_disease_classifier import PlantDiseaseModel, predict_image
import warnings

warnings.filterwarnings("ignore")
# Set page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load model and necessary files


@st.cache_resource
def load_model_resources():
    # Load model configuration
    with open("models/model_config.json", "r") as f:
        config = json.load(f)

    # Load class names
    with open(config["class_names_path"], "r") as f:
        class_names = json.load(f)

    # Load label encoder
    with open(config["label_encoder_path"], "rb") as f:
        label_encoder = pickle.load(f)

    # Load image transformation
    with open(config["transform_path"], "rb") as f:
        transform = pickle.load(f)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlantDiseaseModel(num_classes=len(class_names))
    model.load_state_dict(torch.load(
        config["model_path"], map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    return model, transform, label_encoder, class_names, device, config

# Function to make prediction


def predict(image_file, model, transform, label_encoder, device):
    # Save uploaded file temporarily
    with open("temp_upload.jpg", "wb") as f:
        f.write(image_file.getvalue())

    # Make prediction
    class_name, confidence, all_probs = predict_image(
        model, "temp_upload.jpg", transform, device, label_encoder
    )

    # Get top 5 predictions
    class_indices = np.argsort(all_probs)[::-1][:5]
    top_classes = [label_encoder.inverse_transform(
        [idx])[0] for idx in class_indices]

    # Format the class names for display
    formatted_top_classes = [format_class_name(
        class_name) for class_name in top_classes]
    formatted_primary_class = format_class_name(class_name)

    top_probabilities = [all_probs[idx] * 100 for idx in class_indices]

    # Remove temporary file
    os.remove("temp_upload.jpg")

    # Return both original and formatted class names
    return class_name, formatted_primary_class, confidence, top_classes, formatted_top_classes, top_probabilities


# Function to display the prediction

def display_prediction(original_class, formatted_class, confidence, top_classes, formatted_top_classes, top_probabilities):
    # Display prediction
    st.subheader("Prediction:")
    st.markdown(
        f"<h3 style='color: #4CAF50;'>Diagnosis: {formatted_class}</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<h4>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)

    # Display top 5 predictions
    st.subheader("Top 5 Predictions:")

    # Create DataFrame for top predictions
    prediction_df = pd.DataFrame({
        "Disease": formatted_top_classes,
        "Confidence": top_probabilities
    })

    # Display as a bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(prediction_df["Disease"],
                   prediction_df["Confidence"], color='green')
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Disease")
    ax.set_title("Top 5 Predictions")

    # Add percentage labels
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{prediction_df['Confidence'][i]:.2f}%",
                va='center')

    st.pyplot(fig)

    # Display as a table
    st.table(prediction_df.style.format({"Confidence": "{:.2f}%"}))


def format_class_name(name):
    return name.replace("_", " ").title().replace("  ", " ").replace("  ", " ")

def display_disease_info(class_name, class_names):
    disease_info = {
        "Pepper__bell___Bacterial_spot": {
            "description": "Bacterial spot causes dark, water-soaked lesions on leaves and fruit, leading to defoliation and reduced yield.",
            "causes": "Caused by Xanthomonas campestris pv. vesicatoria, often spread through contaminated seeds and splashing water.",
            "treatment": "Apply copper-based bactericides and remove infected plant parts.",
            "prevention": "Use disease-free seeds, avoid overhead watering, and ensure good air circulation."
        },
        "Pepper__bell___healthy": {
            "description": "Healthy bell pepper plants exhibit firm stems, dark green leaves, and robust fruit development.",
            "causes": "Optimal growth conditions with proper nutrition, watering, and pest control.",
            "treatment": "Maintain proper plant care and monitoring to prevent diseases.",
            "prevention": "Regular fertilization, proper spacing, and pest management."
        },
        "Potato___Early_blight": {
            "description": "Early blight results in brown, concentric ring lesions on leaves, leading to defoliation and reduced tuber quality.",
            "causes": "Caused by Alternaria solani, thriving in warm, humid conditions.",
            "treatment": "Apply fungicides like chlorothalonil or mancozeb and remove infected foliage.",
            "prevention": "Rotate crops, ensure proper spacing, and use resistant potato varieties."
        },
        "Potato___Late_blight": {
            "description": "Late blight causes dark, water-soaked lesions on leaves and tubers, leading to rapid decay.",
            "causes": "Caused by Phytophthora infestans, favored by cool, moist conditions.",
            "treatment": "Use fungicides like metalaxyl and promptly remove infected plants.",
            "prevention": "Plant resistant varieties, ensure good drainage, and avoid overhead watering."
        },
        "Potato___healthy": {
            "description": "Healthy potato plants have lush green foliage, strong stems, and well-developed tubers.",
            "causes": "Proper soil preparation, watering, and pest management.",
            "treatment": "Maintain regular care and disease monitoring.",
            "prevention": "Practice crop rotation, ensure balanced fertilization, and control pests."
        },
        "Tomato_Bacterial_spot": {
            "description": "Bacterial spot causes small, dark lesions on leaves and fruit, reducing yield and quality.",
            "causes": "Caused by Xanthomonas campestris pv. vesicatoria, spread through contaminated tools and water.",
            "treatment": "Use copper-based sprays and remove infected leaves.",
            "prevention": "Avoid overhead irrigation, sanitize tools, and use disease-free seeds."
        },
        "Tomato_Early_blight": {
            "description": "Early blight leads to brown, concentric ring spots on leaves and stems, weakening the plant.",
            "causes": "Caused by Alternaria solani, thriving in warm, humid conditions.",
            "treatment": "Apply fungicides and remove affected plant parts.",
            "prevention": "Practice crop rotation, ensure proper plant spacing, and use resistant varieties."
        },
        "Tomato_Late_blight": {
            "description": "Late blight causes water-soaked lesions on leaves and fruit, leading to rapid plant decline.",
            "causes": "Caused by Phytophthora infestans, spreading in cool, wet conditions.",
            "treatment": "Use fungicides like metalaxyl and remove infected plants.",
            "prevention": "Avoid overhead watering, increase air circulation, and plant resistant varieties."
        },
        "Tomato_Leaf_Mold": {
            "description": "Leaf mold appears as yellow spots on leaves, leading to reduced photosynthesis and yield loss.",
            "causes": "Caused by Passalora fulva, thriving in high humidity.",
            "treatment": "Apply fungicides and improve air circulation.",
            "prevention": "Ensure proper spacing, prune excess foliage, and avoid overhead watering."
        },
        "Tomato_Septoria_leaf_spot": {
            "description": "Septoria leaf spot causes small, dark lesions with a yellow halo, leading to premature leaf drop.",
            "causes": "Caused by Septoria lycopersici, thriving in wet conditions.",
            "treatment": "Use fungicides and remove infected leaves.",
            "prevention": "Practice crop rotation, ensure good air circulation, and water at the base."
        },
        "Tomato_Spider_mites_Two_spotted_spider_mite": {
            "description": "Spider mites cause yellowing and stippling on leaves, leading to plant weakening.",
            "causes": "Caused by Tetranychus urticae, thriving in hot, dry conditions.",
            "treatment": "Use insecticidal soap or neem oil.",
            "prevention": "Regularly mist plants, introduce natural predators like ladybugs, and avoid drought stress."
        },
        "Tomato__Target_Spot": {
            "description": "Target spot appears as dark, concentric lesions on leaves and stems, weakening the plant.",
            "causes": "Caused by Corynespora cassiicola, spreading in humid environments.",
            "treatment": "Apply fungicides and remove infected plant parts.",
            "prevention": "Ensure proper spacing, prune excess foliage, and maintain dry foliage."
        },
        "Tomato__Tomato_YellowLeaf__Curl_Virus": {
            "description": "This viral disease causes yellowing and curling of leaves, leading to stunted growth.",
            "causes": "Spread by whiteflies.",
            "treatment": "No direct cure; manage whitefly populations with insecticides and resistant varieties.",
            "prevention": "Use reflective mulches, introduce natural predators, and remove infected plants."
        },
        "Tomato__Tomato_mosaic_virus": {
            "description": "Mosaic virus causes mottled, distorted leaves and reduced fruit yield.",
            "causes": "Spread through infected seeds, tools, and human handling.",
            "treatment": "No cure; remove infected plants and sanitize tools.",
            "prevention": "Use virus-free seeds, wash hands before handling plants, and control insect vectors."
        },
        "Tomato_healthy": {
            "description": "Healthy tomato plants show vibrant green leaves, strong stems, and normal fruit development.",
            "causes": "Proper care, adequate watering, good sunlight exposure, and regular fertilization.",
            "treatment": "Continue regular care practices to maintain plant health.",
            "prevention": "Regular monitoring, balanced nutrition, appropriate watering, and good air circulation."
        }
    }

    if class_name in disease_info:
        st.subheader("Disease Information:")
        info = disease_info[class_name]

        st.markdown("#### Description")
        st.write(info["description"])

        st.markdown("#### Causes")
        st.write(info["causes"])

        st.markdown("#### Treatment")
        st.write(info["treatment"])

        st.markdown("#### Prevention")
        st.write(info["prevention"])
    else:
        st.info("Detailed information for this specific plant condition is not available. Please consult with an agricultural expert.")


def main():
    try:
        model, transform, label_encoder, class_names, device, config = load_model_resources()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False

    # App title and header
    st.title("🌿 Plant Disease Classifier")
    st.markdown("""
    This application uses deep learning to diagnose diseases in plant leaves. 
    Simply upload an image of a plant leaf, and the model will predict if it's healthy or identify the disease.
    """)

    # Create two columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Image")
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png"])

        # Example images
        st.subheader("Or try an example:")
        example_container = st.container()
        with example_container:
            example_col1, example_col2, example_col3 = st.columns(3)

            if example_col1.button("Healthy Tomato"):
                uploaded_file = "images/examples/tomato_healthy.jpg"
            if example_col2.button("Potato Late blight"):
                uploaded_file = "images/examples/Potato_Late_blight.jpeg"
            if example_col3.button("Pepper bell Bacterial spot"):
                uploaded_file = "images/examples/Pepper_bell_Bacterial_spot.jpeg"

        # Display available classes
        with st.expander("Available Plant Diseases for Classification"):
            # Format class names for display (replace underscores with spaces)
            formatted_classes = [name.replace(
                "_", " ").replace("__", " ").replace("___", " ") for name in class_names]
            # Display in multiple columns for better use of space
            columns = st.columns(3)
            for i, class_name in enumerate(formatted_classes):
                columns[i % 3].markdown(f"- {class_name}")

    with col2:
        st.subheader("Results")
        # Display the uploaded image and prediction results
        if uploaded_file is not None:
            # If it's a string, it's an example image path
            if isinstance(uploaded_file, str):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image",
                         use_container_width=True)
                with open(uploaded_file, "rb") as f:
                    file_content = f.read()
                    uploaded_file_obj = type('obj', (object,), {
                        'getvalue': lambda: file_content
                    })
                class_name, formatted_class, confidence, top_classes, formatted_top_classes, top_probabilities = predict(
                    uploaded_file_obj, model, transform, label_encoder, device
                )
            else:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image",
                         use_container_width=True)
                class_name, formatted_class, confidence, top_classes, formatted_top_classes, top_probabilities = predict(
                    uploaded_file_obj, model, transform, label_encoder, device
                )

            display_prediction(class_name, formatted_class, confidence,
                               top_classes, formatted_top_classes, top_probabilities)


            display_disease_info(class_name, class_names)

    # Show model information
    with st.sidebar:
        st.header("About the Model")
        st.write(
            "This application uses a Convolutional Neural Network (CNN) to classify plant diseases from leaf images.")

        st.subheader("Model Architecture:")
        if model_loaded:
            st.write("- **Model Type:** CNN with 5 convolutional blocks with advanced architectures")
            st.write(f"- **Number of Classes:** {len(class_names)}")


            # Add model performance metrics
            st.write("- **Test Accuracy:** 96.5%")
            
        st.subheader("Dataset Information:")
        st.write("- Trained on the PlantVillage dataset")
        st.write("- Contains 15 classes of plant diseases and healthy plants")
        st.write("- Classes include diseases in tomatoes, potatoes, and peppers")

        st.subheader("Usage Instructions:")
        st.write("1. Upload an image of a plant leaf")
        st.write("2. View the diagnosis and recommended actions")
        st.write("3. Check detailed information about the disease")

        st.subheader("Developers:")
        st.markdown("""
        - **Sayed Gamal**
        - **Youssef Mohammed**
        """)

        st.subheader("Project Repository:")
        st.markdown(
            "[GitHub: Plant-Disease-Classifier](https://github.com/sayedgamal99/Plant-Disease-Classifier)")

        st.markdown("---")
        st.caption("© 2025 Plant Disease Classifier - All Rights Reserved")


if __name__ == "__main__":
    main()
