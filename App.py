import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import time
import matplotlib.pyplot as plt
import io

# Function to predict digit
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    img = np.array(image, dtype='float32') / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result, pred[0]

# Streamlit Page Config
st.set_page_config(page_title='Handwritten Digit Recognition', layout='wide')

st.markdown("""
    <style>
        body {
            background-color: #FFF;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #FFF;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #FF7F50 ;
        }
    </style>
""", unsafe_allow_html=True)

# UI Layout
st.markdown("<div class='title'>Handwritten Digit Recognition</div>", unsafe_allow_html=True)
st.write("### Draw a digit below and click 'Predict'!")

col1, col2 = st.columns([3, 1])

# Create a session state to store the canvas key
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas_1"  # Unique key for the canvas

with col1:
    # Enlarged Canvas without default buttons
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=15,
        stroke_color="#FFF",
        background_color="#000000 ",
        height=400,  # Increased height
        width=633,   # Increased width
        key=st.session_state.canvas_key,  # Dynamic key
        update_streamlit=True,
        drawing_mode="freedraw",
        display_toolbar=False,  # Removes default canvas buttons
    )

with col2:
    if st.button('Predict', use_container_width=True):
        if canvas_result.image_data is not None:
            st.text("Processing...")
            time.sleep(1)
            input_image = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
            res, probabilities = predictDigit(input_image)
            st.success(f'Predicted Digit: {res}', icon="✅")

            # Plot pie chart for probability distribution
            fig, ax = plt.subplots()
            labels = [str(i) for i in range(10)]
            ax.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig)
        else:
            st.warning('Please draw a digit first!', icon="⚠️")

    # Buttons under "Predict"
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("Clear Canvas", use_container_width=True):
            st.session_state.canvas_key = f"canvas_{time.time()}"  # Assign a new key
            st.rerun()  # Only updates the canvas, not the sidebar

    with col_btn2:
        if canvas_result.image_data is not None:
            img_bytes = io.BytesIO()
            input_image = Image.fromarray((canvas_result.image_data[:, :, 0:3]).astype('uint8'))
            input_image.save(img_bytes, format="PNG")
            st.download_button("Download Image", img_bytes.getvalue(), "digit.png", "image/png", use_container_width=True)

# Custom CSS for Sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #1e1e2e;
            color: white;
            padding: 20px;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
            color: #FF7F50;
        }
        [data-testid="stSidebar"] a {
            color: #ff4b4b;
            font-weight: bold;
            text-decoration: none;
        }
        [data-testid="stSidebar"] a:hover {
            color: #FF7F50;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Content
st.sidebar.title("🖥️ How It Works?")
st.sidebar.write("""
1️⃣ **Draw** a digit (0-9) on the canvas  
2️⃣ Click **Predict** to recognize it  
3️⃣ View the predicted digit and probability chart  
4️⃣ Download your drawing if needed  
""")

st.sidebar.title("About Developer 📢")
st.sidebar.write("""
👩‍💻 **Komalpreet Kaur**  
🔗 [GitHub](https://github.com/komalpreetkaurr)  
📧 Email: [kpreetk.879@gmail.com](mailto:kpreetk.879@gmail.com)  
🌍 Location: India  
""")
