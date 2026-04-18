import io
import json
import os
import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from PyPDF2 import PdfReader


load_dotenv()

class EducationEntry(BaseModel):
    degree: str | None = None
    institution: str | None = None
    years: str | None = None
    cgpa: str | None = None


class ResumeSchema(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience_summary: str | None = None
    education: list[EducationEntry] = Field(default_factory=list)


SKILL_ALIASES: dict[str, str] = {
    "machine learning basics": "machine learning",
    "ml": "machine learning",
    "numpy": "numpy",
    "pandas": "pandas",
    "js": "javascript",
    "py": "python",
    "gcp": "google cloud",
    "aws": "amazon web services",
    "airflow": "apache airflow",
    "pytorch": "torch",
    "tensorflow": "tensorflow",
}


def clean_json_text(text: Any) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    fenced = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    fenced_any = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if fenced_any:
        return fenced_any.group(1).strip()

    return text


def extract_text_from_pdf(uploaded_file) -> str:
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [p.extract_text() or "" for p in reader.pages]
    text = "\n\n".join(pages)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_skill(skill: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(skill).lower())
    text = re.sub(r"\s+", " ", text).strip()
    return SKILL_ALIASES.get(text, text)


def skills_match(candidate_skill: str, required_skill: str, threshold: float = 0.86) -> bool:
    candidate = normalize_skill(candidate_skill)
    required = normalize_skill(required_skill)

    if not candidate or not required:
        return False

    if candidate == required:
        return True
    if candidate in required or required in candidate:
        return True

    ratio = SequenceMatcher(None, candidate, required).ratio()
    return ratio >= threshold


def count_skill_matches(candidate_skills: list[str], required_skills: list[str]) -> int:
    if not candidate_skills or not required_skills:
        return 0

    used_candidate_indices: set[int] = set()
    matches = 0

    for req in required_skills:
        matched = False
        for idx, cand in enumerate(candidate_skills):
            if idx in used_candidate_indices:
                continue
            if skills_match(cand, req):
                used_candidate_indices.add(idx)
                matches += 1
                matched = True
                break
        if not matched:
            continue

    return matches


def calculate_similarity(candidate_skills: list[str], required_skills: list[str]) -> float:
    if not required_skills:
        return 0.0

    candidate_list = [s for s in candidate_skills if str(s).strip()]
    required_list = [s for s in required_skills if str(s).strip()]
    if not required_list:
        return 0.0

    intersection = count_skill_matches(candidate_list, required_list)
    union = len(candidate_list) + len(required_list) - intersection
    return round((intersection / union) * 100, 2) if union else 0.0


def calculate_coverage(candidate_skills: list[str], required_skills: list[str]) -> float:
    if not required_skills:
        return 0.0

    candidate_list = [s for s in candidate_skills if str(s).strip()]
    required_list = [s for s in required_skills if str(s).strip()]
    if not required_list:
        return 0.0

    matched = count_skill_matches(candidate_list, required_list)
    return round((matched / len(required_list)) * 100, 2)


def calculate_composite_score(candidate_skills: list[str], required_skills: list[str], llm_score: Any) -> float:
    coverage = calculate_coverage(candidate_skills, required_skills)
    similarity = calculate_similarity(candidate_skills, required_skills)

    # Blend deterministic skill-match metrics with LLM signal so rankings are not flat.
    llm_numeric = pd.to_numeric(pd.Series([llm_score]), errors="coerce").iloc[0]
    if pd.isna(llm_numeric):
        llm_numeric = 0.0

    composite = (0.55 * coverage) + (0.35 * similarity) + (0.10 * float(llm_numeric))
    return round(float(composite), 2)


def calculate_resume_match_score(candidate_skills: list[str], required_skills: list[str]) -> float:
    coverage = calculate_coverage(candidate_skills, required_skills)
    similarity = calculate_similarity(candidate_skills, required_skills)
    return round((0.7 * coverage) + (0.3 * similarity), 2)


def build_chains(llm: ChatOpenAI):
    parser = PydanticOutputParser(pydantic_object=ResumeSchema)

    prompt_extract = PromptTemplate(
        input_variables=["resume_text"],
        template="""
You are an expert resume parser. Extract the following fields and return ONLY valid JSON:
name, email, phone, skills(list), experience_summary, education(list)

Resume:
{resume_text}
""",
    )
    extract_chain = prompt_extract | llm

    skill_gap_prompt = PromptTemplate(
        input_variables=["candidate_skills", "required_skills"],
        template="""
Candidate skills: {candidate_skills}
Required skills: {required_skills}

Return ONLY JSON with:
- missing_skills (list)
- recommendations (object: skill -> recommendation)
""",
    )
    skill_gap_chain = skill_gap_prompt | llm

    final_prompt = PromptTemplate(
        input_variables=["candidate_json", "skill_gap_json"],
        template="""
Generate a final candidate report with:
name, email, phone, skills, missing_skills, recommendations,
experience_summary, education, overall_score(0-100), short_recommendation.

Candidate JSON:
{candidate_json}

Skill Gap JSON:
{skill_gap_json}

Return ONLY JSON.
""",
    )
    final_chain = final_prompt | llm

    jd_extraction_prompt = PromptTemplate(
        input_variables=["job_description"],
        template="""
Extract ALL technical skills mentioned in this job description.
Return ONLY a JSON array of skill names as strings.

Job Description:
{job_description}

Return format: ["skill1", "skill2", ...]
""",
    )
    jd_extract_chain = jd_extraction_prompt | llm

    return parser, extract_chain, skill_gap_chain, final_chain, jd_extract_chain


def extract_required_skills(jd_extract_chain, job_description: str) -> list[str]:
    jd_msg = jd_extract_chain.invoke({"job_description": job_description})
    jd_text = clean_json_text(jd_msg.content)
    try:
        skills = json.loads(jd_text)
        if isinstance(skills, list):
            return [str(s).strip() for s in skills if str(s).strip()]
    except Exception:
        pass

    return ["Python", "SQL", "AWS", "Pandas", "Machine Learning", "Docker"]


def process_resume(
    resume_text: str,
    required_skills: list[str],
    parser: PydanticOutputParser,
    extract_chain,
    skill_gap_chain,
    final_chain,
):
    errors: list[str] = []

    extract_msg = extract_chain.invoke({"resume_text": resume_text[:4000]})
    extract_result = clean_json_text(extract_msg.content)

    parsed = None
    try:
        parsed = parser.parse(extract_result)
    except Exception as exc:
        errors.append(f"Extraction parsing failed: {exc}")

    candidate_skills = parsed.skills if parsed and parsed.skills else []

    sg_msg = skill_gap_chain.invoke(
        {
            "candidate_skills": ", ".join(candidate_skills),
            "required_skills": ", ".join(required_skills),
        }
    )
    sg_text = clean_json_text(sg_msg.content)

    skill_gap_json: dict[str, Any] = {}
    try:
        skill_gap_json = json.loads(sg_text)
        if not isinstance(skill_gap_json, dict):
            skill_gap_json = {}
    except Exception:
        errors.append("Skill gap JSON parsing failed")

    candidate_json_text = parsed.model_dump_json() if parsed else "{}"
    skill_gap_json_text = json.dumps(skill_gap_json) if skill_gap_json else sg_text

    final_msg = final_chain.invoke(
        {
            "candidate_json": candidate_json_text,
            "skill_gap_json": skill_gap_json_text,
        }
    )
    final_result = clean_json_text(final_msg.content)

    final_report: dict[str, Any]
    try:
        final_report = json.loads(final_result)
        if not isinstance(final_report, dict):
            final_report = {"raw": final_result}
    except Exception:
        errors.append("Final report JSON parsing failed")
        final_report = {"raw": final_result}

    if parsed and "skills" not in final_report:
        final_report["skills"] = parsed.skills

    if parsed:
        final_report.setdefault("name", parsed.name)
        final_report.setdefault("email", parsed.email)
        final_report.setdefault("phone", parsed.phone)
        final_report.setdefault("experience_summary", parsed.experience_summary)
        final_report.setdefault("education", [entry.model_dump() for entry in parsed.education])

    # Keep the raw model score for debugging, but use a deterministic score in the table.
    final_report["model_overall_score"] = final_report.get("overall_score")
    final_report["overall_score"] = calculate_resume_match_score(candidate_skills, required_skills)

    return final_report, skill_gap_json, errors


def build_ranking_dataframe(reports: list[dict[str, Any]], required_skills: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for report in reports:
        candidate_skills = report.get("skills", [])
        if not isinstance(candidate_skills, list):
            candidate_skills = []

        similarity = calculate_similarity(candidate_skills, required_skills)
        coverage = calculate_coverage(candidate_skills, required_skills)
        llm_score = report.get("overall_score", 0)
        composite_score = calculate_composite_score(candidate_skills, required_skills, llm_score)
        rows.append(
            {
                "Name": report.get("name", "N/A"),
                "Email": report.get("email", "N/A"),
                "Phone": report.get("phone", "N/A"),
                "Skills": ", ".join(candidate_skills),
                "Score": composite_score,
                "LLM Score": llm_score,
                "Coverage %": coverage,
                "Similarity %": similarity,
                "Missing Skills": ", ".join(report.get("missing_skills", [])),
                "Recommendation": report.get("short_recommendation", "N/A"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0)
    df["LLM Score"] = pd.to_numeric(df["LLM Score"], errors="coerce").fillna(0)
    df = df.sort_values(by=["Score", "Similarity %"], ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def report_identity(report: dict[str, Any]) -> tuple[str, str, str]:
    email = normalize_text(report.get("email"))
    phone = normalize_text(report.get("phone"))
    name = normalize_text(report.get("name"))
    return email, phone, name


def deduplicate_reports(reports: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    dropped = 0

    for report in reports:
        identity = report_identity(report)

        # If all identifying fields are missing, keep the row to avoid dropping valid edge cases.
        if not any(identity):
            deduped.append(report)
            continue

        if identity in seen:
            dropped += 1
            continue

        seen.add(identity)
        deduped.append(report)

    return deduped, dropped


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

        :root {
            --bg-soft: #f7faf8;
            --ink: #10221a;
            --muted: #456558;
            --brand: #0f766e;
            --accent: #d97706;
            --card: #ffffff;
            --border: #d9e8e1;
        }

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 5% 10%, #d8f3ea 0, rgba(216,243,234,0) 35%),
                radial-gradient(circle at 95% 0%, #fff4dd 0, rgba(255,244,221,0) 28%),
                var(--bg-soft);
        }

        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #0f766e 0%, #1f9d90 52%, #f59e0b 100%);
            color: #ffffff;
            box-shadow: 0 10px 26px rgba(16,34,26,0.14);
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: 0.2px;
        }

        .hero p {
            margin: 0.45rem 0 0;
            font-size: 1rem;
            opacity: 0.96;
        }

        .panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            box-shadow: 0 6px 20px rgba(16,34,26,0.05);
        }

        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0d4f48;
            margin: 0.3rem 0 0.6rem;
        }

        .skill-chip {
            display: inline-block;
            background: #eaf6f2;
            color: #0e4f49;
            border: 1px solid #c8e9df;
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            margin: 0.2rem 0.28rem 0.2rem 0;
            font-size: 0.84rem;
            font-weight: 600;
        }

        .leader-card {
            background: linear-gradient(180deg, #ffffff 0%, #f5fbf8 100%);
            border: 1px solid #d9ece4;
            border-radius: 14px;
            padding: 0.8rem;
            box-shadow: 0 4px 14px rgba(16,34,26,0.05);
            min-height: 112px;
        }

        .leader-rank {
            font-size: 0.8rem;
            color: #5d7f71;
            margin: 0;
        }

        .leader-name {
            font-size: 1.02rem;
            font-weight: 800;
            margin: 0.1rem 0 0.4rem;
            color: #12342a;
        }

        .leader-score {
            font-size: 0.92rem;
            color: #214e3d;
            margin: 0.1rem 0;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_skill_chips(skills: list[str]) -> None:
    if not skills:
        st.info("No required skills extracted from the job description.")
        return

    chip_html = "".join(f"<span class='skill-chip'>{skill}</span>" for skill in skills)
    st.markdown(chip_html, unsafe_allow_html=True)


def render_top_candidate_cards(df: pd.DataFrame) -> None:
    if df.empty:
        return

    st.markdown("<p class='section-title'>Top Matches</p>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i in range(3):
        if i >= len(df):
            break

        row = df.iloc[i]
        with cols[i]:
            st.markdown(
                (
                    "<div class='leader-card'>"
                    f"<p class='leader-rank'>Rank #{int(row['Rank'])}</p>"
                    f"<p class='leader-name'>{row['Name']}</p>"
                    f"<p class='leader-score'>Composite Score: {float(row['Score']):.2f}</p>"
                    f"<p class='leader-score'>Similarity: {float(row['Similarity %']):.2f}%</p>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def main():
    st.set_page_config(page_title="AI Resume Analyzer", page_icon="🎯", layout="wide")
    inject_custom_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>AI Resume Analyzer</h1>
            <p>Upload resumes, compare against a role, and spotlight top candidates live.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Configuration")
        env_key = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=env_key,
            help="Uses .env key by default if present.",
        )
        model_name = st.text_input("Model", value="gpt-4o-mini")

    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        st.markdown("<p class='section-title'>Job Description</p>", unsafe_allow_html=True)
        job_description = st.text_area(
            "Job Description",
            value="",
            placeholder="Paste the job description here...",
            height=315,
            label_visibility="collapsed",
        )

    with right_col:
        st.markdown("<p class='section-title'>Resume Upload</p>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload one or more resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        uploaded_count = len(uploaded_files) if uploaded_files else 0
        st.caption(f"{uploaded_count} file(s) selected")
        run_analysis = st.button("Analyze Resumes", type="primary", width="stretch")

    if run_analysis:
        if not api_key:
            st.error("OPENAI API key is required. Add it in the sidebar or .env.")
            return
        if not uploaded_files:
            st.error("Please upload at least one resume PDF.")
            return
        if not job_description.strip():
            st.error("Please type or paste a job description before analyzing resumes.")
            return

        llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        parser, extract_chain, skill_gap_chain, final_chain, jd_extract_chain = build_chains(llm)

        with st.spinner("Extracting required skills from job description..."):
            required_skills = extract_required_skills(jd_extract_chain, job_description)

        st.markdown("<p class='section-title'>Required Skills</p>", unsafe_allow_html=True)
        render_skill_chips(required_skills)

        reports: list[dict[str, Any]] = []
        logs: list[str] = []

        progress = st.progress(0)
        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            try:
                uploaded_file.seek(0)
                resume_text = extract_text_from_pdf(uploaded_file)
                if not resume_text:
                    logs.append(f"{uploaded_file.name}: No extractable text found.")
                    continue

                final_report, _skill_gap, errors = process_resume(
                    resume_text=resume_text,
                    required_skills=required_skills,
                    parser=parser,
                    extract_chain=extract_chain,
                    skill_gap_chain=skill_gap_chain,
                    final_chain=final_chain,
                )
                final_report["_filename"] = uploaded_file.name
                reports.append(final_report)
                for err in errors:
                    logs.append(f"{uploaded_file.name}: {err}")
            except Exception as exc:
                logs.append(f"{uploaded_file.name}: Processing failed - {exc}")

            progress.progress(idx / len(uploaded_files))

        if not reports:
            st.warning("No resumes were successfully processed.")
            if logs:
                st.subheader("Logs")
                for msg in logs:
                    st.write(f"- {msg}")
            return

        reports, dropped_count = deduplicate_reports(reports)
        if dropped_count:
            logs.append(f"Removed {dropped_count} duplicate candidate report(s) based on name/email/phone.")

        df = build_ranking_dataframe(reports, required_skills)

        render_top_candidate_cards(df)

        st.markdown("<p class='section-title'>Candidate Ranking</p>", unsafe_allow_html=True)
        st.dataframe(df, width="stretch")

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Ranking CSV",
            data=csv_data,
            file_name="candidate_ranking.csv",
            mime="text/csv",
            width="stretch",
        )

        st.markdown("<p class='section-title'>Per-Candidate Reports</p>", unsafe_allow_html=True)
        for idx, report in enumerate(reports, start=1):
            label = report.get("name") or report.get("_filename", "Candidate")
            with st.expander(str(label), expanded=False):
                st.json(report)
                json_bytes = json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
                safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", str(label)).strip("_") or "candidate"
                unique_suffix = report.get("_filename") or str(idx)
                safe_suffix = re.sub(r"[^A-Za-z0-9_-]", "_", str(unique_suffix)).strip("_") or str(idx)
                st.download_button(
                    f"Download JSON - {label}",
                    data=json_bytes,
                    file_name=f"{safe_name}_{safe_suffix}.json",
                    mime="application/json",
                    key=f"download_{safe_name}_{safe_suffix}_{idx}",
                    width="stretch",
                )

        if logs:
            st.subheader("Logs")
            for msg in logs:
                st.write(f"- {msg}")


if __name__ == "__main__":
    main()
