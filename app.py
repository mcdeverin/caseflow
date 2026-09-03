import json
import re

import streamlit as st
from pypdf import PdfReader
from anthropic import Anthropic
from pyairtable import Api
from pyairtable.formulas import match

st.set_page_config(
    page_title="CaseFlow AI",
    page_icon="📄",
    layout="wide"
)
st.title("CaseFlow AI")
st.subheader("AI-powered workers' comp claim intake")

client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Two tables now, not one. Table names hardcoded rather than pulled from a
# secret — they're structural, not sensitive. Assumed "Documents" and
# "Cases" based on your CSV export filenames; tell me if either is wrong.
_api = Api(st.secrets["AIRTABLE_TOKEN"])
documents_table = _api.table(st.secrets["AIRTABLE_BASE_ID"], "Documents")
cases_table = _api.table(st.secrets["AIRTABLE_BASE_ID"], "Cases")

# --- Two tiers of "required," on purpose --------------------------------
# A single document in a claim packet (a witness statement, a timecard) is
# not expected to carry every field. Only the CASE, across all its linked
# documents, needs to be complete.
#
# PER_DOCUMENT_REQUIRED — checked here, before saving. Just enough to
# identify and match this document to a case.
#
# CASE_LEVEL_REQUIRED — not enforced yet. Target for the case-rollup /
# Airtable automation step, not built yet.

PER_DOCUMENT_REQUIRED = ["employee_name", "incident_date"]

CASE_LEVEL_REQUIRED = [
    "incident_location",
    "incident_description",
    "injury_type",
    "body_area",
    "supervisor_name",
]

# Document types this tool is actually scoped to. Anything else — Claude
# classifies as "Other" — doesn't belong in a workers' comp pipeline at all,
# and shouldn't be asked to satisfy PER_DOCUMENT_REQUIRED (an incident_date
# requirement makes no sense for a document that was never about an
# incident, like a benefits dependent-verification form).
CLAIMS_DOCUMENT_TYPES = {"Incident Report", "Medical Provider Note", "Witness Statement", "Timecard"}


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
You are an AI document intake system for a workers' compensation claims
process. The document you're given is one piece of a claim packet, not the
whole claim. It may be:
- An incident report
- A medical provider note
- A witness statement
- A timecard relevant to the incident (e.g. hours worked that day)
- Another supporting document

Return ONLY valid JSON using this structure:
{{
    "document_type": null,
    "employee_name": null,
    "employer": null,
    "incident_date": null,
    "incident_location": null,
    "incident_description": null,
    "injury_type": null,
    "body_area": null,
    "supervisor_name": null,
    "summary": null,
    "recommended_action": null
}}

Instructions:
1. Identify which of these document types this is — use this exact casing:
   "Incident Report", "Medical Provider Note", "Witness Statement",
   "Timecard", or "Other". Do not invent a different label or casing.
2. Extract employee name and employer when present.
3. Extract the incident date this document relates to, when present.
4. Extract incident_location, incident_description, injury_type, body_area,
   and supervisor_name only when THIS document actually states them. A
   witness statement or timecard may legitimately have most of these as
   null — that is expected, not a failure.
5. Write a concise factual summary of the document.
6. Suggest one brief recommended next action for a human reviewer.

Rules:
- Never invent information.
- If a field is absent from this specific document, use null.
- Return JSON only. Do not include markdown formatting.
<document>
{text}
</document>
"""
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}],
    )

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
    return [f for f in PER_DOCUMENT_REQUIRED if not data.get(f)]


def find_matching_cases(data):
    """Exact match on Employee Name (+ Employer, if we have one). Not fuzzy —
    a real version would need to handle typos/nicknames, but exact is the
    honest starting point rather than guessing at a similarity threshold."""
    employee_name = (data.get("employee_name") or "").strip()
    if not employee_name:
        return []
    criteria = {"Employee Name": employee_name}
    employer = (data.get("employer") or "").strip()
    if employer:
        criteria["Employer"] = employer
    return cases_table.all(formula=match(criteria))


def find_conflicts(data, case_record):
    """Compares this document's incident_date against what's already on the
    matched case. Deliberately limited — injury_type, body_area, etc. only
    exist on Documents, not rolled up to a Case-level column, so they can't
    be compared this way without a schema change. Also note: no date-format
    normalization here, so 'August 11, 2026' vs '2026-08-11' would falsely
    read as a conflict even when they're the same date — a real version
    would need to parse both into a common format first."""
    conflicts = []
    new_date = (data.get("incident_date") or "").strip()
    existing_date = str(case_record.get("fields", {}).get("Incident Date") or "").strip()
    if new_date and existing_date and new_date != existing_date:
        conflicts.append(f"Incident Date — this document says \"{new_date}\", case has \"{existing_date}\"")
    return conflicts


def find_duplicate_document(case_record, filename):
    """Checks the matched case's already-linked documents for one with the
    same filename. Returns that document's record id, or None."""
    linked_doc_ids = case_record.get("fields", {}).get("Documents", [])
    for doc_id in linked_doc_ids:
        doc = documents_table.get(doc_id)
        for attachment in doc.get("fields", {}).get("Document", []):
            if attachment.get("filename") == filename:
                return doc_id
    return None


def create_case(data):
    # AI Case Summary and AI Recommended Action are left out on purpose —
    # they read as native Airtable AI fields (computed from other fields in
    # the record, same idea as the autonumber Case ID), which the API can't
    # set directly. If they're configured to reference the linked Documents,
    # Airtable computes them itself once a document is linked.
    fields = {
        "Employee Name": data.get("employee_name"),
        "Employer": data.get("employer"),
        "Claim Type": "Workers Compensation",
        "Incident Date": data.get("incident_date"),
        "Incident Description": data.get("incident_description") or data.get("summary"),
        "Case Status": "New",
    }
    return cases_table.create(fields, typecast=True)


def attach_pdf_to_document(doc_id, uploaded_file):
    documents_table.upload_attachment(
        doc_id,
        "Document",
        filename=uploaded_file.name,
        content=uploaded_file.getvalue(),
        content_type="application/pdf",
    )


def create_document_record(data, extracted_text, missing, case_id):
    fields = {
        "Document Type": data.get("document_type"),
        "Extraction Status": "Complete" if not missing else "Needs Review",
        "Extracted Text": extracted_text,
        "Extracted Data": json.dumps(data),
        "Extraction Notes": f"Missing: {', '.join(missing)}" if missing else "",
        "Extraction summary": data.get("summary"),
        "Key entities extracted": f"{data.get('employee_name') or '—'} / {data.get('employer') or '—'}",
        "Case": [case_id],
    }
    return documents_table.create(fields, typecast=True)


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

        if file_key not in st.session_state:
            st.session_state[file_key] = {"data": None, "reviewed": False, "saved": False}

        try:
            extracted_text = extract_pdf_text(uploaded_file)
            if not extracted_text.strip():
                st.warning("No readable text was found in this document.")
                continue

            st.success("Text extracted successfully")
            with st.expander("View extracted text"):
                st.text(extracted_text)

            if st.button("Analyze Claim", key=f"analyze_{file_key}"):
                with st.spinner("Analyzing document..."):
                    st.session_state[file_key]["data"] = analyze_document(extracted_text)
                    st.session_state[file_key]["reviewed"] = False
                    st.session_state[file_key]["saved"] = False
                    st.session_state[file_key].pop("matches", None)

            data = st.session_state[file_key]["data"]

            if data:
                with st.expander("View extracted data"):
                    st.json(data)

                if data.get("document_type") not in CLAIMS_DOCUMENT_TYPES:
                    st.error(
                        f"This doesn't look like a workers' comp claims document "
                        f"(classified as: {data.get('document_type') or 'Other'}). "
                        "This tool is scoped to Claims intake only — "
                        f"{data.get('recommended_action') or 'route it to the appropriate team instead.'}"
                    )
                    continue

                missing = check_completeness(data)

                if not st.session_state[file_key]["reviewed"]:
                    if missing:
                        st.warning(
                            f"Missing to identify/match this document: {', '.join(missing)}. "
                            "Review and complete below."
                        )
                        with st.form(key=f"edit_form_{file_key}"):
                            employee_name = st.text_input(
                                "Employee Name" + (" ⚠️" if "employee_name" in missing else ""),
                                value=data.get("employee_name") or "",
                            )
                            incident_date = st.text_input(
                                "Incident Date" + (" ⚠️" if "incident_date" in missing else ""),
                                value=data.get("incident_date") or "",
                            )
                            submitted = st.form_submit_button("Confirm corrections")
                        if submitted:
                            data["employee_name"] = employee_name
                            data["incident_date"] = incident_date
                            st.session_state[file_key]["data"] = data
                            st.session_state[file_key]["reviewed"] = True
                            st.rerun()
                    else:
                        st.session_state[file_key]["reviewed"] = True

                if st.session_state[file_key]["reviewed"] and not st.session_state[file_key]["saved"]:
                    # Resolve the case match once, cache it — this can hit the
                    # API and (for the "no match" case) creates nothing yet,
                    # so it's safe to compute on every rerun, but caching
                    # avoids redundant calls while the person is deciding.
                    if "matches" not in st.session_state[file_key]:
                        st.session_state[file_key]["matches"] = find_matching_cases(data)

                    matches = st.session_state[file_key]["matches"]
                    case_choice_id = None

                    if len(matches) == 1:
                        case_fields = matches[0]["fields"]
                        case_display = case_fields.get("Case ID", matches[0]["id"])
                        emp = case_fields.get("Employee Name", "?")
                        employer = case_fields.get("Employer", "")
                        conflicts = find_conflicts(data, matches[0])

                        st.markdown(f"**Found existing claim: {case_display}**")
                        st.write(f"- Employee: {emp}")
                        st.write(f"- Employer: {employer or '—'}")
                        st.write(f"- Incident Date on file: {case_fields.get('Incident Date', '—')}")

                        attach_label = f"Attach to existing claim {case_display}"
                        choice = st.radio(
                            "What should happen with this document?",
                            [attach_label, "This is a different claim — create a new one"],
                            key=f"case_choice_{file_key}",
                        )
                        case_choice_id = matches[0]["id"] if choice == attach_label else "NEW"

                    elif len(matches) == 0:
                        case_choice_id = "NEW"
                        st.info("No existing claim found for this employee — will create a new one.")

                    else:
                        st.warning(f"{len(matches)} existing claims match this employee — pick the right one.")
                        options = {
                            f"{m['fields'].get('Employee Name', '?')} — "
                            f"{m['fields'].get('Employer', '?')} (…{m['id'][-6:]})": m["id"]
                            for m in matches
                        }
                        options["None of these — create a new claim"] = "NEW"
                        label = st.selectbox(
                            "Which claim is this document for?",
                            list(options.keys()),
                            key=f"case_pick_{file_key}",
                        )
                        case_choice_id = options[label]

                    if st.button("Save to Airtable", key=f"save_{file_key}"):
                        if case_choice_id == "NEW":
                            case_id = create_case(data)["id"]
                            dup_id = None
                        else:
                            case_id = case_choice_id
                            case_record = cases_table.get(case_id)
                            dup_id = find_duplicate_document(case_record, uploaded_file.name)

                        if dup_id:
                            st.error(
                                f"{uploaded_file.name} already appears to be linked to this "
                                "case — not saving a duplicate."
                            )
                        else:
                            doc_record = create_document_record(data, extracted_text, missing, case_id)
                            attach_pdf_to_document(doc_record["id"], uploaded_file)
                            st.session_state[file_key]["saved"] = True
                            st.rerun()

                if st.session_state[file_key]["saved"]:
                    st.success("Saved to Airtable.")

        except Exception as e:
            st.error(f"Processing failed: {e}")
else:
    st.info("Upload one or more PDF documents to begin.")
