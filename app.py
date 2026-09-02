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

    response_text = message.content[0].text.strip()

    if response_text.startswith("```"):
        response_text = response_text.replace("```json", "", 1)
        response_text = response_text.replace("```", "")
        response_text = response_text.strip()

    if not response_text:
        raise ValueError("AI returned an empty response.")

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise ValueError(
            f"AI returned a response that was not valid JSON: {response_text}"
        )
