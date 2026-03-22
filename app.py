import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 🔹 Load Model
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return {"model": model, "tokenizer": tokenizer}

model_data = load_model()
model = model_data["model"]
tokenizer = model_data["tokenizer"]

# -------------------------------
# 🔹 UI
# -------------------------------
st.set_page_config(page_title="Smart Study Assistant")

st.title("📚 Smart Study Assistant")
st.write("Upload a PDF and get AI summary + smart Q&A (Improved RAG)")

# -------------------------------
# 🔹 Upload PDF
# -------------------------------
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

    # -------------------------------
    # 🔹 SUMMARY SECTION
    # -------------------------------
    st.subheader("🧠 Generate Summary")

    if st.button("Generate AI Summary"):
        with st.spinner("⏳ Generating summary..."):
            try:
                input_text = text[:2000]

                if len(input_text.strip()) < 50:
                    st.error("❌ PDF text too short.")
                else:
                    inputs = tokenizer(
                        input_text,
                        max_length=512,
                        truncation=True,
                        return_tensors="pt"
                    )

                    summary_ids = model.generate(
                        inputs["input_ids"],
                        max_length=150,
                        min_length=50,
                        num_beams=4,
                        early_stopping=True
                    )

                    summary_text = tokenizer.decode(
                        summary_ids[0],
                        skip_special_tokens=True
                    )

                    st.write(summary_text)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # -------------------------------
    # 🔹 RAG SECTION (IMPROVED)
    # -------------------------------
    st.subheader("❓ Ask Questions from PDF")

    # 🔥 Better chunking (by size, not lines)
    chunk_size = 300
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    query = st.text_input("Enter your question")

    if query:
        with st.spinner("🔍 Finding best answer..."):
            try:
                # Vectorization
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(chunks + [query])

                # Similarity
                similarity = cosine_similarity(vectors[-1], vectors[:-1])

                # 🔥 Get top 3 relevant chunks
                top_indices = similarity[0].argsort()[-3:][::-1]

                context = ""
                for i in top_indices:
                    context += chunks[i] + "\n"

                st.subheader("📄 Relevant Context")
                st.write(context)

                # 🔥 Improved prompt
                prompt = f"""
You are a helpful study assistant.

Answer ONLY using the context below.
If the answer is not present, say "Not found in document".

Explain clearly and simply.

Context:
{context}

Question:
{query}
"""

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                output_ids = model.generate(
                    inputs["input_ids"],
                    max_length=200,
                    num_beams=4,
                    early_stopping=True
                )

                answer = tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                )

                st.subheader("🧠 Answer")
                st.write(answer)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")