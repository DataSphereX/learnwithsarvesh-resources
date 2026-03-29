from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ResumeSchema(BaseModel):
    name: str = Field(default="Unknown")
    email: str = Field(default="")
    phone: str = Field(default="")
    skills: List[str] = Field(default_factory=list)
    experience_years: float = Field(default=0.0)
    summary: str = Field(default="")


class ResumeResult(BaseModel):
    file_name: str
    parsed_resume: ResumeSchema
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    match_score: float = Field(default=0.0)
    report: str = Field(default="")
