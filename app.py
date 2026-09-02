import streamlit as st

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

    for file in uploaded_files:
        st.write(f"• {file.name}")
else:
    st.info("Upload one or more PDF documents to begin.")
