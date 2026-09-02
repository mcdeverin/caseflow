import streamlit as st
from pypdf import PdfReader

st.set_page_config(
    page_title="CaseFlow AI",
    page_icon="📄",
    layout="wide"
)

st.title("CaseFlow AI")
st.subheader("AI-powered claims intake and validation")

uploaded_files = st.file_uploader(
    "Upload claim documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} document(s) uploaded")

    for uploaded_file in uploaded_files:
        st.markdown(f"### {uploaded_file.name}")

        try:
            reader = PdfReader(uploaded_file)

            extracted_text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    extracted_text += page_text + "\n"

            if extracted_text.strip():
                st.success("Text extracted successfully")

                with st.expander("View extracted text"):
                    st.text(extracted_text)

            else:
                st.warning("No readable text was found in this document.")

        except Exception as e:
            st.error(f"Extraction failed: {e}")

else:
    st.info("Upload one or more PDF documents to begin.")
