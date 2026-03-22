import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load summarization model (will download once)
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return {"model": model, "tokenizer": tokenizer}

model_data = load_model()
model = model_data["model"]
tokenizer = model_data["tokenizer"]

# UI
st.set_page_config(page_title="Smart Study Assistant")

st.title("📚 Smart Study Assistant")
st.write("Upload a PDF and get AI-powered summary (FREE)")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    pdf = PdfReader(uploaded_file)

    text = ""
    for page in pdf.pages:
        if page.extract_text():
            text += page.extract_text()

    st.success("✅ PDF loaded successfully!")

    # Preview
    st.subheader("📄 Preview")
    st.write(text[:1000])

    # Generate Summary
    if st.button("Generate AI Summary"):
        with st.spinner("⏳ Generating summary..."):

            try:
                # Limit input size to model's max tokens
                input_text = text[:2048]
                
                if len(input_text.strip()) < 50:
                    st.error("❌ PDF text is too short for summarization. Please upload a PDF with more content.")
                else:
                    # Tokenize and summarize
                    inputs = tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt")
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        max_length=150,
                        min_length=50,
                        num_beams=4,
                        early_stopping=True
                    )
                    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

                    st.subheader("🧠 AI Summary")
                    st.write(summary_text)

            except Exception as e:
                st.error(f"❌ Error generating summary: {str(e)}")