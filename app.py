import streamlit as st
import pickle
import docx2txt
import PyPDF2

# -----------------------------
# Load Model, Vectorizer, and LabelEncoder
# -----------------------------
@st.cache_resource
def load_model():
    with open("decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfv = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tfv, le

model, tfv, le = load_model()

# -----------------------------
# Helper: Extract text from file
# -----------------------------
def extract_text(file):
    text = ""
    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8", errors="ignore")
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        st.error("❌ Unsupported file format. Please upload .txt, .docx, or .pdf")
    return text

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📄 Resume Classification (Decision Tree)")

st.write("Upload a resume file or paste text manually to classify.")

# Option 1: Text area
resume_text = st.text_area("Paste Resume Text Here:")

# Option 2: File upload
uploaded_file = st.file_uploader("Or upload a file (.txt, .docx, .pdf)", type=["txt", "docx", "pdf"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)

if st.button("Predict"):
    if resume_text.strip():
        vectorized_text = tfv.transform([resume_text])
        pred_num = model.predict(vectorized_text)[0]
        pred_label = le.inverse_transform([pred_num])[0]   # convert back to category name
        st.success(f"✅ Predicted Category: **{pred_label}**")
    else:
        st.warning("⚠️ Please provide resume text or upload a file.")
