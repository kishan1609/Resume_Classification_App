import streamlit as st
import pickle
import re
import docx  # for .docx support

# Load pipeline (TF-IDF + Decision Tree)
with open("resume_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit UI
st.set_page_config(page_title="Resume Classification", page_icon="📄", layout="centered")

st.title("🌳 Resume Classification Project")
st.write("This app classifies resumes into categories using a **Decision Tree model** pipeline.")

# Input method
option = st.radio("Choose input method:", ("Paste Resume Text", "Upload .txt File", "Upload .docx File"))

resume_text = ""
if option == "Paste Resume Text":
    resume_text = st.text_area("Enter Resume Text Below:", height=200)

elif option == "Upload .txt File":
    uploaded_file = st.file_uploader("Upload a resume (.txt format only)", type=["txt"])
    if uploaded_file:
        resume_text = uploaded_file.read().decode("utf-8")

elif option == "Upload .docx File":
    uploaded_file = st.file_uploader("Upload a resume (.docx format only)", type=["docx"])
    if uploaded_file:
        doc = docx.Document(uploaded_file)
        resume_text = " ".join([para.text for para in doc.paragraphs])

# Prediction button
if st.button("🔍 Classify Resume"):
    if resume_text.strip() != "":
        clean_text = preprocess_text(resume_text)
        prediction = model.predict([clean_text])[0]
        st.success(f"✅ Predicted Category: **{prediction}**")
    else:
        st.warning("⚠️ Please enter or upload some resume text.")

st.markdown("---")
st.caption("🚀 Deployed with Streamlit | Model: Decision Tree Classifier Pipeline")
