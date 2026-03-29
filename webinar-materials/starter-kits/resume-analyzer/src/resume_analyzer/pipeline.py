from __future__ import annotations

import json
import re
from typing import Iterable

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from .config import AppConfig
from .schemas import ResumeResult, ResumeSchema


def _clean_json_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^```(?:json)?", "", value).strip()
    value = re.sub(r"```$", "", value).strip()
    return value


def _normalize_skills(skills: Iterable[str]) -> set[str]:
    return {skill.strip().lower() for skill in skills if skill and skill.strip()}


class ResumeAnalyzer:
    def __init__(self, config: AppConfig) -> None:
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=0,
            api_key=config.openai_api_key,
        )

        self.extract_prompt = PromptTemplate.from_template(
            """
You are an expert recruiter. Extract structured candidate information from resume text.
Return strict JSON only with keys:
name, email, phone, skills (list), experience_years (number), summary.

Resume:
{resume_text}
""".strip()
        )

        self.jd_skills_prompt = PromptTemplate.from_template(
            """
Extract only technical skills from the job description.
Return strict JSON in this format: {"skills": ["skill1", "skill2"]}

Job Description:
{job_description}
""".strip()
        )

        self.report_prompt = PromptTemplate.from_template(
            """
You are a hiring assistant. Write a concise evaluation in 5-8 bullet points.
Include: strengths, skill gaps, role fit, and recommendation.

Candidate JSON:
{candidate_json}

Matched Skills:
{matched_skills}

Missing Skills:
{missing_skills}
""".strip()
        )

    def extract_required_skills(self, job_description: str) -> list[str]:
        response = self.llm.invoke(self.jd_skills_prompt.format(job_description=job_description))
        payload = json.loads(_clean_json_text(response.content))
        skills = payload.get("skills", [])
        return sorted({str(item).strip() for item in skills if str(item).strip()})

    def process_resume(
        self,
        file_name: str,
        resume_text: str,
        required_skills: list[str],
    ) -> ResumeResult:
        extract_response = self.llm.invoke(
            self.extract_prompt.format(resume_text=resume_text[:9000])
        )
        parsed = ResumeSchema.model_validate_json(_clean_json_text(extract_response.content))

        candidate_skills = _normalize_skills(parsed.skills)
        required = _normalize_skills(required_skills)

        matched = sorted(required.intersection(candidate_skills))
        missing = sorted(required.difference(candidate_skills))
        score = (len(matched) / max(1, len(required))) * 100

        report_response = self.llm.invoke(
            self.report_prompt.format(
                candidate_json=parsed.model_dump_json(indent=2),
                matched_skills=", ".join(matched) or "None",
                missing_skills=", ".join(missing) or "None",
            )
        )

        return ResumeResult(
            file_name=file_name,
            parsed_resume=parsed,
            matched_skills=matched,
            missing_skills=missing,
            match_score=round(score, 2),
            report=report_response.content.strip(),
        )
