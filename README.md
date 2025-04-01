# Handwritten Digit Recognition

## Overview
This project is a **Handwritten Digit Recognition** system that classifies handwritten digits (0-9) using a **Convolutional Neural Network (CNN)**. The model is trained in a **Jupyter Notebook (IPYNB file)**, and the interface is built using **Streamlit**. The project is deployed online using **Render**.

🔗 **Live Demo:** [Handwritten Digit Recognition](https://handwrittendigitreco.onrender.com)

## Features
- Classifies handwritten digits from 0 to 9
- Built with **TensorFlow/Keras, NumPy, Pandas, Matplotlib, Seaborn**
- Uses **Streamlit** for an interactive UI
- Allows users to **draw digits on a canvas** for real-time prediction
- Displays a **pie chart visualization** of model confidence scores
- Trained using **Convolutional Neural Networks (CNNs)**
- Deployed online via **Render**
- Supports real-time digit prediction

## Technologies Used
- **Python**
- **TensorFlow/Keras**
- **NumPy, Pandas, Matplotlib, Seaborn**
- **Streamlit**
- **Jupyter Notebook**
- **Render** (for deployment)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app locally:
   ```bash
   streamlit run app.py
   ```

## Dataset
- The **MNIST dataset** is used for training and testing.
- It consists of **70,000 images (28x28 pixels) of handwritten digits**.

## Model Training
- Implemented in **Jupyter Notebook** (`Main_model.ipynb`).
- Uses a **CNN architecture** for high accuracy.
- Data is preprocessed using **Normalization and Reshaping**.
- Performance evaluated using **accuracy score**.

## Usage
1. Open the **live demo** [here](https://handwrittendigitreco.onrender.com).
2. Draw a digit on the canvas.
3. The model predicts the digit and displays the result.
4. A **pie chart** shows the model's confidence scores for each digit.
   
## License
This project is licensed under the **MIT License**.
