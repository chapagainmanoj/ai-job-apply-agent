import re
from enum import Enum
from datetime import date
from typing import Literal, Optional, List, Dict, Any, TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field, field_validator, EmailStr, HttpUrl
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# Enums for better type safety
class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    SOFT = "soft"
    DOMAIN_SPECIFIC = "domain_specific"
    LANGUAGE = "language"
    CERTIFICATION = "certification"


class ProficiencyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class RequirementImportance(str, Enum):
    REQUIRED = "required"
    PREFERRED = "preferred"
    NICE_TO_HAVE = "nice_to_have"


class JobType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"


class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    CERTIFICATE = "certificate"


# Pydantic models for structured data
class PersonalInfo(BaseModel):
    name: str = Field(..., description="Full name")
    email: Optional[EmailStr] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Current location")
    linkedin_url: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    portfolio_url: Optional[HttpUrl] = Field(None, description="Portfolio website URL")

    @field_validator("phone")
    def validate_phone(cls, v):
        if v and not re.match(r"^[\+]?[\d\s\-\(\)\.]{10,}$", v):
            raise ValueError("Invalid phone number format")
        return v


class Skill(BaseModel):
    name: str = Field(..., description="Skill name")
    category: SkillCategory = Field(..., description="Skill category")
    proficiency_level: ProficiencyLevel = Field(..., description="Proficiency level")
    years_experience: Optional[int] = Field(None, ge=0, le=50, description="Years of experience")
    last_used: Optional[date] = Field(None, description="When skill was last used")
    context: Optional[str] = Field(None, description="Context where skill was used")

    class Config:
        use_enum_values = True


class WorkExperience(BaseModel):
    company: str = Field(..., description="Company name")
    position: str = Field(..., description="Job position/title")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date or 'Present'")
    duration_months: int = Field(..., ge=0, description="Duration in months")
    location: Optional[str] = Field(None, description="Work location")
    employment_type: Optional[JobType] = Field(None, description="Employment type")
    responsibilities: List[str] = Field(default_factory=list, description="Job responsibilities")
    achievements: List[str] = Field(default_factory=list, description="Key achievements")
    skills_used: List[str] = Field(default_factory=list, description="Skills utilized in role")
    skills_learned: List[str] = Field(default_factory=list, description="Skills acquired in role")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")

    @field_validator("end_date")
    def validate_end_date(cls, v, values):
        if v.lower() not in ["present", "current"] and "start_date" in values:
            # Add date validation logic here
            pass
        return v

    class Config:
        use_enum_values = True


class Education(BaseModel):
    institution: str = Field(..., description="Educational institution")
    degree: str = Field(..., description="Degree name")
    field_of_study: str = Field(..., description="Field of study")
    level: Optional[EducationLevel] = Field(None, description="Education level")
    graduation_date: Optional[str] = Field(None, description="Graduation date")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA score")
    relevant_coursework: List[str] = Field(default_factory=list, description="Relevant courses")
    honors: List[str] = Field(default_factory=list, description="Academic honors")
    activities: List[str] = Field(default_factory=list, description="Academic activities")

    class Config:
        use_enum_values = True


class Project(BaseModel):
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    url: Optional[HttpUrl] = Field(None, description="Project URL")
    github_url: Optional[HttpUrl] = Field(None, description="GitHub repository URL")
    start_date: Optional[str] = Field(None, description="Project start date")
    end_date: Optional[str] = Field(None, description="Project end date")
    role: Optional[str] = Field(None, description="Your role in the project")
    achievements: List[str] = Field(default_factory=list, description="Project achievements")


class Certification(BaseModel):
    name: str = Field(..., description="Certification name")
    issuing_organization: str = Field(..., description="Issuing organization")
    issue_date: Optional[str] = Field(None, description="Issue date")
    expiry_date: Optional[str] = Field(None, description="Expiry date")
    credential_id: Optional[str] = Field(None, description="Credential ID")
    credential_url: Optional[HttpUrl] = Field(None, description="Verification URL")


class Language(BaseModel):
    name: str = Field(..., description="Language name")
    proficiency: Literal["native", "fluent", "conversational", "basic"] = Field(..., description="Proficiency level")
    certification: Optional[str] = Field(None, description="Language certification")


class ParsedResume(BaseModel):
    personal_info: PersonalInfo = Field(..., description="Personal information")
    professional_summary: str = Field("", description="Professional summary")
    work_experiences: List[WorkExperience] = Field(default_factory=list, description="Work experiences")
    education: List[Education] = Field(default_factory=list, description="Educational background")
    skills: List[Skill] = Field(default_factory=list, description="Skills and competencies")
    certifications: List[Certification] = Field(default_factory=list, description="Certifications")
    projects: List[Project] = Field(default_factory=list, description="Notable projects")
    languages: List[Language] = Field(default_factory=list, description="Language skills")
    volunteer_experience: List[str] = Field(default_factory=list, description="Volunteer work")
    publications: List[str] = Field(default_factory=list, description="Publications")
    awards: List[str] = Field(default_factory=list, description="Awards and recognitions")

    def get_total_experience_years(self) -> float:
        """Calculate total years of work experience"""
        total_months = sum(exp.duration_months for exp in self.work_experiences)
        return round(total_months / 12, 1)

    def get_skills_by_category(self, category: SkillCategory) -> List[Skill]:
        """Get skills filtered by category"""
        return [skill for skill in self.skills if skill.category == category]

    def get_recent_experience(self, years: int = 5) -> List[WorkExperience]:
        """Get work experience from recent years"""
        # Simplified - in practice, would parse dates properly
        return self.work_experiences[:years]


class JobRequirement(BaseModel):
    skill: str = Field(..., description="Required skill")
    importance: RequirementImportance = Field(..., description="Requirement importance")
    category: SkillCategory = Field(..., description="Skill category")
    years_required: Optional[int] = Field(None, ge=0, description="Years of experience required")
    description: Optional[str] = Field(None, description="Detailed requirement description")

    class Config:
        use_enum_values = True


class CompanyInfo(BaseModel):
    name: str = Field(..., description="Company name")
    size: Optional[str] = Field(None, description="Company size")
    location: Optional[str] = Field(None, description="Company location")
    website: Optional[HttpUrl] = Field(None, description="Company website")
    description: Optional[str] = Field(None, description="Company description")


class ParsedJobDescription(BaseModel):
    company: CompanyInfo = Field(..., description="Company information")
    position: str = Field(..., description="Job position title")
    location: str = Field(..., description="Job location")
    job_type: JobType = Field(..., description="Employment type")
    salary_range: Optional[str] = Field(None, description="Salary range")
    remote_options: bool = Field(False, description="Remote work available")
    summary: str = Field("", description="Job summary")
    responsibilities: List[str] = Field(default_factory=list, description="Job responsibilities")
    requirements: List[JobRequirement] = Field(default_factory=list, description="Job requirements")
    preferred_qualifications: List[str] = Field(default_factory=list, description="Preferred qualifications")
    benefits: List[str] = Field(default_factory=list, description="Benefits offered")
    company_culture: List[str] = Field(default_factory=list, description="Company culture aspects")
    application_deadline: Optional[str] = Field(None, description="Application deadline")

    class Config:
        use_enum_values = True

    def get_required_skills(self) -> List[JobRequirement]:
        """Get only required skills"""
        return [req for req in self.requirements if req.importance == RequirementImportance.REQUIRED]

    def get_preferred_skills(self) -> List[JobRequirement]:
        """Get preferred skills"""
        return [req for req in self.requirements if req.importance == RequirementImportance.PREFERRED]


class SkillMatch(BaseModel):
    requirement: JobRequirement = Field(..., description="Job requirement")
    resume_skill: Optional[Skill] = Field(None, description="Matching resume skill")
    match_strength: float = Field(..., ge=0.0, le=1.0, description="Match strength (0-1)")
    gap_description: Optional[str] = Field(None, description="Description of skill gap")


class SkillMatchAnalysis(BaseModel):
    overall_match_score: float = Field(..., ge=0.0, le=100.0, description="Overall match percentage")
    required_skills_match_score: float = Field(..., ge=0.0, le=100.0, description="Required skills match percentage")
    matched_requirements: List[SkillMatch] = Field(default_factory=list, description="Matched requirements")
    unmatched_requirements: List[JobRequirement] = Field(default_factory=list, description="Unmatched requirements")
    skill_gaps: List[JobRequirement] = Field(default_factory=list, description="Critical skill gaps")
    transferable_skills: List[Skill] = Field(default_factory=list, description="Transferable skills")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")

    def is_good_match(self, threshold: float = 70.0) -> bool:
        """Check if this is a good match based on threshold"""
        return self.overall_match_score >= threshold


class CoverLetter(BaseModel):
    header: str = Field(..., description="Cover letter header")
    opening_paragraph: str = Field(..., description="Opening paragraph")
    body_paragraphs: List[str] = Field(..., description="Body paragraphs")
    closing_paragraph: str = Field(..., description="Closing paragraph")
    signature: str = Field(..., description="Letter signature")

    def get_full_letter(self) -> str:
        """Get the complete cover letter"""
        body = "\n\n".join(self.body_paragraphs)
        return f"{self.header}\n\n{self.opening_paragraph}\n\n{body}\n\n{self.closing_paragraph}\n\n{self.signature}"

    def get_word_count(self) -> int:
        """Get word count of the cover letter"""
        full_text = self.get_full_letter()
        return len(full_text.split())


class RecruiterQuestion(BaseModel):
    question: str = Field(..., description="Recruiter question")
    category: Literal["experience", "motivation", "skills", "salary", "availability", "general"] = Field(
        ..., description="Question category"
    )
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence level")

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if answer has high confidence"""
        return self.confidence >= threshold


# State management for LangGraph - Base state without messages
class ApplicationState(TypedDict):
    resume_text: str
    job_description_text: str
    parsed_resume: Optional[ParsedResume]
    parsed_job_description: Optional[ParsedJobDescription]
    skill_match_analysis: Optional[SkillMatchAnalysis]
    cover_letter: Optional[CoverLetter]
    recruiter_questions: Optional[List[str]]
    recruiter_answers: Optional[List[RecruiterQuestion]]
    current_step: str
    errors: List[str]


# Extended state for LangGraph with message handling
class LangGraphApplicationState(ApplicationState):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class ApplicationResult(BaseModel):
    """Final result of the application process"""

    parsed_resume: Optional[ParsedResume] = None
    parsed_job_description: Optional[ParsedJobDescription] = None
    skill_match_analysis: Optional[SkillMatchAnalysis] = None
    cover_letter: Optional[CoverLetter] = None
    recruiter_answers: Optional[List[RecruiterQuestion]] = None
    errors: List[str] = Field(default_factory=list)
    status: str = "completed"
    processing_time_seconds: Optional[float] = None

    def export_to_json(self) -> str:
        """Export results to JSON"""
        return self.model_dump_json(exclude_none=True, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the results"""
        summary = {
            "status": self.status,
            "errors_count": len(self.errors),
            "has_cover_letter": self.cover_letter is not None,
            "recruiter_questions_answered": len(self.recruiter_answers) if self.recruiter_answers else 0,
        }

        if self.skill_match_analysis:
            summary.update(
                {
                    "overall_match_score": self.skill_match_analysis.overall_match_score,
                    "required_skills_match": self.skill_match_analysis.required_skills_match_score,
                    "is_good_match": self.skill_match_analysis.is_good_match(),
                }
            )

        if self.parsed_resume:
            summary.update(
                {
                    "total_experience_years": self.parsed_resume.get_total_experience_years(),
                    "skills_count": len(self.parsed_resume.skills),
                    "work_experiences_count": len(self.parsed_resume.work_experiences),
                }
            )

        return summary
