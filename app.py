import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import time

# ----------------------------
# CONSTANTS
# ----------------------------
IMG_SIZE = (224, 224)
TUMOR_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

st.title("🧠 MRI Classification & Brain Tumor Detection")
st.write("Upload a brain scan image. The app first checks if it is an MRI. If yes, it performs tumor classification.")

st.write(f"TensorFlow version: {tf.__version__}")

# ----------------------------
# LOAD MODELS (CACHED)
# ----------------------------
@st.cache_resource
def load_mri_binary_model():
    # STEP 1 MODEL: MRI vs Non-MRI
    return tf.keras.models.load_model("EfficientNet_MRI_v_NonMRI.keras")

@st.cache_resource
def load_multiclass_model():
    # STEP 2 MODEL: 4-way classification
    return tf.keras.models.load_model("MobileNetSVM_multiclassification.keras")

try:
    mri_binary_model = load_mri_binary_model()
    tumor_multiclass_model = load_multiclass_model()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# ----------------------------
# IMAGE PREPROCESSING
# ----------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ----------------------------
# UI – IMAGE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("📁 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_input = preprocess(image)

    with st.spinner("🔍 Step 1: Checking if image is MRI..."):
        time.sleep(1.0)
        try:
            mri_pred = mri_binary_model.predict(img_input)
            # Assume output → probability of NON-MRI
            non_mri_prob = float(mri_pred[0][0])
            mri_prob = 1.0 - non_mri_prob

        except Exception as e:
            st.error(f"MRI prediction error: {e}")
            st.stop()

    # ----------------------------
    # DECISION: MRI OR NOT?
    # ----------------------------
    st.subheader("Step 1 Result: MRI Detection")

    if mri_prob >= 0.5:
        st.success(f"🧲 **This IS an MRI image** (confidence: {mri_prob:.4f})")

        # ----------------------------
        # STEP 2 — MULTICLASS TUMOR DETECTION
        # ----------------------------
        with st.spinner("🧠 Step 2: Classifying tumor type..."):
            time.sleep(1.0)
            try:
                tumor_pred = tumor_multiclass_model.predict(img_input)
                class_index = int(np.argmax(tumor_pred))
                confidence = float(np.max(tumor_pred))
            except Exception as e:
                st.error(f"Tumor classification error: {e}")
                st.stop()

        st.subheader("Step 2 Result: Tumor Classification")
        st.info(f"🧬 Predicted Tumor Type: **{TUMOR_CLASSES[class_index]}**")
        st.write(f"🔎 Confidence: {confidence:.4f}")

    else:
        st.error(f"🚫 **This is NOT an MRI image** (MRI probability = {mri_prob:.4f})")
        st.write("❌ Tumor classification will not be performed.")
