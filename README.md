# Plant Disease Classification System

## Overview
This project is a comprehensive plant disease classification system built from scratch using PyTorch and deployed as a user-friendly web application with Streamlit. The system can identify various diseases in crops like tomatoes, potatoes, and bell peppers through image analysis, providing farmers and gardeners with instant diagnosis and treatment recommendations.

## 🌟 Key Achievements

- **Built a complete CNN model from scratch** for plant disease classification
- **Achieved 96.5% accuracy** on test data for disease identification
- **Created an interactive web application** with Streamlit for easy access
- **Implemented a complete ML pipeline** from data preparation to deployment
- **Designed a user-friendly interface** with diagnostic information and treatment recommendations

## 📊 Features

- **Disease Classification**: Identifies 15+ plant diseases across multiple crops
- **Confidence Metrics**: Provides confidence scores and top 5 predictions
- **Treatment Information**: Offers causes, treatments, and prevention strategies for each disease
- **Interactive Interface**: Allows users to upload their own images or try example images
- **Visualization**: Displays predictions with confidence charts

## 🖥️ Tech Stack

- **Deep Learning**: PyTorch for model development and training
- **Computer Vision**: PIL and torchvision for image processing
- **Web Application**: Streamlit for the user interface
- **Data Science**: NumPy, pandas, and matplotlib for data manipulation and visualization

## 🛠️ Model Architecture

The custom CNN architecture includes:
- 5 convolutional blocks with batch normalization and ReLU activation
- Global average pooling
- Dropout layers to prevent overfitting
- Early stopping and learning rate scheduling for optimal training

## 📋 Dataset

The model was trained on the PlantVillage dataset, which contains thousands of labeled images of healthy and diseased plant leaves across various crop species.

## 📱 Application Workflow

1. User uploads an image of a plant leaf
2. The model processes the image and classifies the disease
3. Results are displayed with confidence scores
4. Detailed information about the disease, causes, and treatments is provided

## 🔧 Installation and Usage

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

## 🧠 Learning Outcomes

- Implementing deep learning models for image classification
- Building an end-to-end machine learning pipeline
- Creating interactive web applications for AI solutions
- Optimizing model training with techniques like early stopping and learning rate scheduling
- Deploying machine learning models in user-friendly applications

## 🔮 Future Improvements

- Expand the dataset to include more plant species and diseases
- Implement transfer learning with pre-trained models for better accuracy
- Add mobile compatibility for in-field diagnosis
- Integrate with weather data to provide contextual prevention advice
- Develop offline functionality for areas with limited internet connectivity

## 📷 Screenshots

[Include screenshots of the application here]

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.