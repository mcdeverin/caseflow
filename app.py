import json
import streamlit as st
from pypdf import PdfReader
from anthropic import Anthropic

st.set_page_config(
    page_title="CaseFlow AI",
    page_icon="📄",
    layout="wide"
)

st.title("CaseFlow AI")
st.subheader("AI-powered claims intake and validation")

client = Anthropic(
    api_key=st.secrets["ANTHROPIC_API_KEY"]
)


def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def analyze_document(text):
    prompt = f"""
You are an AI document intake system for internal operations teams.

Analyze the document and extract structured information.

The document may relate to:
- Workers Compensation
- Benefits
- Payroll
- Other

Return ONLY valid JSON using this structure:

{{
    "document_type": null,
    "claim_type": null,
    "employee_name": null,
    "employer": null,
    "event_date": null,
    "summary": null,
    "attributes": {{}}
}}

Instructions:

1. Identify what type of document this is.
2. Classify the operational area as Workers Compensation, Benefits, Payroll, or Other.
3. Extract employee name and employer when present.
4. Extract the most relevant incident, request, or event date when present.
5. Write a concise factual summary of the document.
6. Put domain-specific information inside "attributes".

Examples of domain-specific attributes include:

Payroll:
- pay_period_start
- pay_period_end
- regular_hours
- overtime_hours
- amount_paid
- manager_approval
- approval_date

Benefits:
- dependent_name
- dependent_date_of_birth
- relationship
- qualifying_event
- coverage_type

Workers Compensation:
- incident_location
- incident_description
- injury_type
- body_area
- supervisor_name

Rules:
- Never invent information.
- If a common field is absent, use null.
- Only include attributes supported by the document.
- Return JSON only.
- Do not include markdown formatting.

<document>
{text}
</document>
"""

    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1200,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response_text = message.content[0].text
    return json.loads(response_text)


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
            extracted_text = extract_pdf_text(uploaded_file)

            if not extracted_text.strip():
                st.warning("No readable text was found in this document.")
                continue

            st.success("Text extracted successfully")

            with st.expander("View extracted text"):
                st.text(extracted_text)

            if st.button(
                "Analyze with AI",
                key=f"analyze_{uploaded_file.name}"
            ):

                with st.spinner("Analyzing document..."):
                    structured_data = analyze_document(extracted_text)

                st.success("AI analysis complete")

                st.subheader("Extracted Data")
                st.json(structured_data)

        except Exception as e:
            st.error(f"Processing failed: {e}")

else:
    st.info("Upload one or more PDF documents to begin.")
