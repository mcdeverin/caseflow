
import json
import re
 
import streamlit as st
from pypdf import PdfReader
from anthropic import Anthropic
from pyairtable import Api
 
st.set_page_config(
    page_title="CaseFlow AI",
    page_icon="📄",
    layout="wide"
)
st.title("CaseFlow AI")
st.subheader("AI-powered claims intake and validation")
 
client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
 
airtable_table = Api(st.secrets["AIRTABLE_TOKEN"]).table(
    st.secrets["AIRTABLE_BASE_ID"],
    st.secrets["AIRTABLE_TABLE_NAME"],
)
 
# --- What "complete" means, per claim type -----------------------------
# These are business-required fields, not just "did the AI find something."
# Adjust to match what your Airtable base / claims process actually needs.
 
TOP_LEVEL_REQUIRED = ["employee_name", "event_date"]
 
REQUIRED_ATTRIBUTES = {
    "Workers Compensation": [
        "incident_location",
        "incident_description",
        "injury_type",
        "body_area",
    ],
    "Benefits": [
        "dependent_name",
        "dependent_date_of_birth",
        "qualifying_event",
        "coverage_type",
    ],
    "Payroll": [
        "pay_period_start",
        "pay_period_end",
        "manager_approval",
        "approval_date",
    ],
}
 
 
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
        messages=[{"role": "user", "content": prompt}],
    )
 
    # Pull text from whatever block actually has it, instead of assuming
    # content[0] is always the text block.
    response_text = "".join(
        block.text for block in message.content if hasattr(block, "text")
    ).strip()
 
    if not response_text:
        raise ValueError(
            f"Claude returned no text. stop_reason={message.stop_reason}, "
            f"content={message.content}"
        )
 
    response_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response_text)
    return json.loads(response_text)
 
 
def check_completeness(data):
    """Returns the list of required fields that are missing. Empty list = complete."""
    missing = [f for f in TOP_LEVEL_REQUIRED if not data.get(f)]
    attributes = data.get("attributes") or {}
    for field in REQUIRED_ATTRIBUTES.get(data.get("claim_type"), []):
        if not attributes.get(field):
            missing.append(field)
    return missing
 
 
def write_to_airtable(data, status, source_filename):
    airtable_table.create({
        "Document Name": source_filename,
        "Document Type": data.get("document_type"),
        "Claim Type": data.get("claim_type"),
        "Employee Name": data.get("employee_name"),
        "Employer": data.get("employer"),
        "Event Date": data.get("event_date"),
        "Summary": data.get("summary"),
        "Attributes (JSON)": json.dumps(data.get("attributes") or {}),
        "Status": status,
    })
 
 
uploaded_files = st.file_uploader(
    "Upload claim documents",
    type=["pdf"],
    accept_multiple_files=True,
)
 
if uploaded_files:
    st.success(f"{len(uploaded_files)} document(s) uploaded")
 
    for uploaded_file in uploaded_files:
        st.markdown(f"### {uploaded_file.name}")
        file_key = uploaded_file.name
 
        # Per-file state so results survive reruns (form submits trigger a
        # full script rerun in Streamlit, and we don't want to lose the
        # analysis or double-write to Airtable when that happens).
        if file_key not in st.session_state:
            st.session_state[file_key] = {"data": None, "saved": False}
 
        try:
            extracted_text = extract_pdf_text(uploaded_file)
            if not extracted_text.strip():
                st.warning("No readable text was found in this document.")
                continue
 
            st.success("Text extracted successfully")
            with st.expander("View extracted text"):
                st.text(extracted_text)
 
            if st.button("Analyze with AI", key=f"analyze_{file_key}"):
                with st.spinner("Analyzing document..."):
                    st.session_state[file_key]["data"] = analyze_document(extracted_text)
                    st.session_state[file_key]["saved"] = False
 
            data = st.session_state[file_key]["data"]
 
            if data:
                st.subheader("Extracted Data")
                st.json(data)
 
                missing = check_completeness(data)
 
                if st.session_state[file_key]["saved"]:
                    st.success("Saved to Airtable.")
 
                elif not missing:
                    # Extraction is complete against required fields —
                    # write straight through. No point making someone
                    # review a claim the AI already got fully.
                    write_to_airtable(data, "AI-Extracted / Unreviewed", file_key)
                    st.session_state[file_key]["saved"] = True
                    st.success("All required fields present — saved to Airtable automatically.")
 
                else:
                    # Something required is missing — stop here and let
                    # the person fix it while the source doc is still in
                    # front of them, instead of finding out later.
                    st.warning(
                        f"Missing required info: {', '.join(missing)}. "
                        "Review and complete below before submitting."
                    )
 
                    with st.form(key=f"edit_form_{file_key}"):
                        employee_name = st.text_input(
                            "Employee Name" + (" ⚠️" if "employee_name" in missing else ""),
                            value=data.get("employee_name") or "",
                        )
                        event_date = st.text_input(
                            "Event Date" + (" ⚠️" if "event_date" in missing else ""),
                            value=data.get("event_date") or "",
                        )
                        attributes = dict(data.get("attributes") or {})
                        for field in REQUIRED_ATTRIBUTES.get(data.get("claim_type"), []):
                            label = field.replace("_", " ").title()
                            if field in missing:
                                label += " ⚠️"
                            attributes[field] = st.text_input(
                                label, value=str(attributes.get(field) or "")
                            )
 
                        submitted = st.form_submit_button("Submit corrected claim")
 
                    if submitted:
                        data["employee_name"] = employee_name
                        data["event_date"] = event_date
                        data["attributes"] = attributes
                        write_to_airtable(data, "Manually Completed", file_key)
                        st.session_state[file_key]["data"] = data
                        st.session_state[file_key]["saved"] = True
                        st.rerun()
 
        except Exception as e:
            st.error(f"Processing failed: {e}")
else:
    st.info("Upload one or more PDF documents to begin.")
 


