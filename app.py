# app.py
import os
import io
import re
import json
import time
import base64
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ---------------------------
# Utils
# ---------------------------

def read_pdf_text(file) -> str:
    """Extracts text from a PDF using PyMuPDF (works on Streamlit Cloud)."""
    try:
        text = []
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                # Extract both layout text and blocks to improve recall
                t = page.get_text("text")
                if not t or t.strip() == "":
                    t = page.get_text("blocks")
                    if isinstance(t, list):
                        t = "\n".join([blk[4] for blk in t if isinstance(blk, (list, tuple)) and len(blk) > 4])
                text.append(t)
        return "\n".join(text)
    except Exception as e:
        st.error(f"Failed reading PDF: {e}")
        return ""

def cleanup_json(s: str) -> str:
    """Strip code fences / stray text and return best-effort JSON string."""
    # Pull the first {...} or [...] block
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.I)
    if fence:
        s = fence.group(1)
    # Keep only outermost JSON object/array
    brace = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", s)
    return brace.group(1) if brace else s.strip()

def groq_extract_structured(cv_text: str, client: Groq, model: str = "llama-3.1-70b-versatile") -> dict:
    """Ask GROQ to extract structured fields from CV text."""
    system = (
        "You are an expert HR resume parser. Extract concise, factual details only "
        "from the provided CV text. If a value is missing, return null."
    )
    schema = {
        "name": None,
        "email": None,
        "phone": None,
        "location_city": None,
        "location_country": None,
        "date_of_birth": None,  # ISO format if possible: YYYY-MM-DD or null
        "years_experience": None,  # number (float or int)
        "current_title": None,
        "current_company": None,
        "highest_qualification": None,  # e.g., MBA, BSc Computer Science
        "universities": [],
        "skills": [],
        "certifications": [],
        "languages": [],
        "notice_period": None,
        "linkedin": None,
        "summary": None
    }

    instructions = (
        "Return a single compact JSON object EXACTLY in this schema and key order:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Constraints:\n"
        "- Only fill fields supported by evidence in the CV text.\n"
        "- Use ISO date format for date_of_birth when possible.\n"
        "- years_experience should be numeric (e.g., 7 or 7.5).\n"
        "- skills should be 5-20 concise items.\n"
        "- Do not add commentary outside JSON."
    )

    user = f"CV TEXT:\n{cv_text[:120000]}"  # keep request size sane

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": instructions},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(cleanup_json(content))
        # Ensure types
        if isinstance(data.get("years_experience"), str):
            # try to parse float from string
            m = re.search(r"\d+(?:\.\d+)?", data["years_experience"])
            data["years_experience"] = float(m.group(0)) if m else None
    except Exception:
        data = {"parse_error": content[:1000]}
    return data

def dataframe_to_pdf(df: pd.DataFrame, title: str = "Candidate Screening Summary") -> bytes:
    """Render a compact PDF report of the summary DataFrame."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=24, bottomMargin=24, leftMargin=24, rightMargin=24)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 12))

    # Convert df to table data
    table_data = [list(df.columns)] + df.astype(str).values.tolist()

    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E6E6E6")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("FONTSIZE", (0,1), (-1,-1), 9),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#BDBDBD")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#BDBDBD")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FAFAFA")]),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))

    doc.build(story)
    buf.seek(0)
    return buf.read()

def to_display_df(records: list[dict]) -> pd.DataFrame:
    """Flatten model output to a clean, presentation-ready DataFrame."""
    rows = []
    for r in records:
        if "parse_error" in r:
            rows.append({"Name": None, "Email": None, "Phone": None, "DOB": None,
                         "Experience (yrs)": None, "Qualification": None, "Location": None,
                         "Current Title": None, "Current Company": None, "Skills (top 8)": None,
                         "Notice": None, "LinkedIn": None, "Status": "LLM parse error"})
            continue

        loc = ", ".join(filter(None, [r.get("location_city"), r.get("location_country")])) or None
        skills = r.get("skills") or []
        if isinstance(skills, list):
            skills_txt = ", ".join(skills[:8])
        else:
            skills_txt = str(skills)

        rows.append({
            "Name": r.get("name"),
            "Email": r.get("email"),
            "Phone": r.get("phone"),
            "DOB": r.get("date_of_birth"),
            "Experience (yrs)": r.get("years_experience"),
            "Qualification": r.get("highest_qualification"),
            "Location": loc,
            "Current Title": r.get("current_title"),
            "Current Company": r.get("current_company"),
            "Skills (top 8)": skills_txt,
            "Notice": r.get("notice_period"),
            "LinkedIn": r.get("linkedin"),
            "Status": "OK"
        })
    df = pd.DataFrame(rows)
    # Order columns
    cols = ["Name","Email","Phone","DOB","Experience (yrs)","Qualification",
            "Location","Current Title","Current Company","Skills (top 8)","Notice","LinkedIn","Status"]
    return df[cols]

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="CV Screening (GROQ)", page_icon="🧠", layout="wide")

st.title("🧠 CV Screening App (GROQ)")
st.caption("Upload candidate CVs in PDF. The app extracts key details via GROQ and produces a summary table + downloadable PDF.")

# Sidebar — API and model
with st.sidebar:
    st.header("Settings")
    default_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    api_key = st.text_input("GROQ API Key", value=default_key, type="password", placeholder="gsk_...")
    model = st.selectbox(
        "Model",
        options=["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192"],
        index=0
    )
    st.markdown("---")
    st.markdown("**Note:** This works best with text-based PDFs. Scanned images require OCR (not included).")

uploaded = st.file_uploader("Upload one or more CVs (PDF)", type=["pdf"], accept_multiple_files=True)

if "results" not in st.session_state:
    st.session_state["results"] = []

col1, col2 = st.columns([1,1])

with col1:
    start_btn = st.button("🔎 Parse CVs", type="primary", use_container_width=True, disabled=not uploaded)
with col2:
    clear_btn = st.button("🧹 Clear", use_container_width=True)

if clear_btn:
    st.session_state["results"] = []
    st.experimental_rerun()

if start_btn:
    if not api_key:
        st.error("Please provide your GROQ API key in the sidebar.")
    else:
        client = Groq(api_key=api_key)
        results = []
        progress = st.progress(0)
        status = st.empty()

        for i, f in enumerate(uploaded, start=1):
            status.info(f"Reading {f.name} ...")
            cv_text = read_pdf_text(f)
            if not cv_text.strip():
                results.append({"parse_error": f"No text extracted from {f.name}."})
                progress.progress(i/len(uploaded))
                continue

            status.info(f"Extracting fields via GROQ for {f.name} ...")
            try:
                data = groq_extract_structured(cv_text, client=client, model=model)
                results.append(data)
            except Exception as e:
                results.append({"parse_error": f"GROQ error: {e}"})

            progress.progress(i/len(uploaded))

        st.session_state["results"] = results
        status.success("Done.")

if st.session_state["results"]:
    df = to_display_df(st.session_state["results"])
    st.subheader("Summary")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Downloads
    pdf_bytes = dataframe_to_pdf(df, title="Candidate Screening Summary")
    st.download_button(
        label="📄 Download Summary PDF",
        data=pdf_bytes,
        file_name="cv_screening_summary.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.download_button(
        label="📥 Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cv_screening_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Raw JSON (per candidate)"):
        st.json(st.session_state["results"])
