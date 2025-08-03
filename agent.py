import re
import json
from typing import Dict, List, Any, Optional, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from datetime import datetime

from schema import (
    ApplicationState,
    ParsedResume,
    ParsedJobDescription,
    SkillMatchAnalysis,
    CoverLetter,
    RecruiterQuestion,
    ApplicationResult,
)


# Updated State class for LangGraph with message handling
class LangGraphApplicationState(ApplicationState):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class ResumeJobApplicationSystem:
    def __init__(self, api_key: str, model: str = "anthropic:claude-3-5-sonnet-latest"):
        """
        Initialize the system with Claude AI integration via LangGraph

        Args:
            api_key: Anthropic API key (set as environment variable)
            model: Claude model to use (default: anthropic:claude-3-5-sonnet-latest)
        """
        # Initialize the LangChain Anthropic model
        self.model = init_chat_model(model, api_key=api_key, timeout=200)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(LangGraphApplicationState)

        # Add nodes
        workflow.add_node("parse_resume", self.parse_resume_node)
        workflow.add_node("parse_job_description", self.parse_job_description_node)
        workflow.add_node("analyze_skill_match", self.analyze_skill_match_node)
        workflow.add_node("generate_cover_letter", self.generate_cover_letter_node)
        workflow.add_node("handle_recruiter_questions", self.handle_recruiter_questions_node)
        workflow.add_node("finalize_application", self.finalize_application_node)

        # Set entry point
        workflow.set_entry_point("parse_resume")

        # Add edges
        workflow.add_edge("parse_resume", "parse_job_description")
        workflow.add_edge("parse_job_description", "analyze_skill_match")
        workflow.add_edge("analyze_skill_match", "generate_cover_letter")
        workflow.add_edge("generate_cover_letter", "handle_recruiter_questions")
        workflow.add_edge("handle_recruiter_questions", "finalize_application")
        workflow.add_edge("finalize_application", END)

        return workflow.compile()

    def _call_claude(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a call to Claude API via LangChain"""
        try:
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=prompt))

            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Claude API error: {e}")
            raise

    def parse_resume_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Parse resume text into structured Pydantic model using Claude"""
        try:
            resume_text = state["resume_text"]
            parsed_resume = self._parse_resume_content_with_claude(resume_text)

            # Validate the parsed resume
            validated_resume = ParsedResume.model_validate(parsed_resume)

            state["parsed_resume"] = validated_resume
            state["current_step"] = "resume_parsed"

            # Add message for tracking
            state["messages"] = [AIMessage(content="✅ Resume parsing completed with Claude AI + Pydantic validation")]
            print("✅ Resume parsing completed with Claude AI + Pydantic validation")

        except Exception as e:
            state["errors"].append(f"Resume parsing error: {str(e)}")
            state["messages"] = [AIMessage(content=f"❌ Resume parsing failed: {e}")]
            print(f"❌ Resume parsing failed: {e}")

        return state

    def parse_job_description_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Parse job description into structured Pydantic model using Claude"""
        try:
            job_desc_text = state["job_description_text"]
            parsed_job = self._parse_job_description_content_with_claude(job_desc_text)

            # Validate the parsed job description
            validated_job = ParsedJobDescription.model_validate(parsed_job)

            state["parsed_job_description"] = validated_job
            state["current_step"] = "job_description_parsed"

            # Update messages
            new_message = AIMessage(content="✅ Job description parsing completed with Claude AI + Pydantic validation")
            state["messages"] = state.get("messages", []) + [new_message]
            print("✅ Job description parsing completed with Claude AI + Pydantic validation")

        except Exception as e:
            state["errors"].append(f"Job description parsing error: {str(e)}")
            error_message = AIMessage(content=f"❌ Job description parsing failed: {e}")
            state["messages"] = state.get("messages", []) + [error_message]
            print(f"❌ Job description parsing failed: {e}")

        return state

    def analyze_skill_match_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Analyze skill matching with Claude AI and Pydantic models"""
        try:
            resume = state["parsed_resume"]
            job = state["parsed_job_description"]

            if not resume or not job:
                raise ValueError("Missing parsed resume or job description")

            skill_analysis = self._analyze_skill_matching_with_claude(resume, job)

            # Validate the analysis
            validated_analysis = SkillMatchAnalysis.model_validate(skill_analysis)

            state["skill_match_analysis"] = validated_analysis
            state["current_step"] = "skill_analysis_completed"

            # Update messages
            new_message = AIMessage(content="✅ Skill matching analysis completed with Claude AI + Pydantic validation")
            state["messages"] = state.get("messages", []) + [new_message]
            print("✅ Skill matching analysis completed with Claude AI + Pydantic validation")

        except Exception as e:
            state["errors"].append(f"Skill analysis error: {str(e)}")
            error_message = AIMessage(content=f"❌ Skill analysis failed: {e}")
            state["messages"] = state.get("messages", []) + [error_message]
            print(f"❌ Skill analysis failed: {e}")

        return state

    def generate_cover_letter_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Generate tailored cover letter with Claude AI and Pydantic model"""
        try:
            resume = state["parsed_resume"]
            job = state["parsed_job_description"]
            skill_analysis = state["skill_match_analysis"]

            if not all([resume, job, skill_analysis]):
                raise ValueError("Missing required data for cover letter generation")

            cover_letter = self._generate_cover_letter_with_claude(resume, job, skill_analysis)

            # Validate the cover letter
            validated_cover_letter = CoverLetter.model_validate(cover_letter)

            state["cover_letter"] = validated_cover_letter
            state["current_step"] = "cover_letter_generated"

            # Update messages
            new_message = AIMessage(content="✅ Cover letter generated with Claude AI + Pydantic validation")
            state["messages"] = state.get("messages", []) + [new_message]
            print("✅ Cover letter generated with Claude AI + Pydantic validation")

        except Exception as e:
            state["errors"].append(f"Cover letter generation error: {str(e)}")
            error_message = AIMessage(content=f"❌ Cover letter generation failed: {e}")
            state["messages"] = state.get("messages", []) + [error_message]
            print(f"❌ Cover letter generation failed: {e}")

        return state

    def handle_recruiter_questions_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Handle recruiter questions with Claude AI and Pydantic models"""
        try:
            if state.get("recruiter_questions"):
                resume = state["parsed_resume"]
                job = state["parsed_job_description"]
                questions = state["recruiter_questions"]

                answers = self._answer_recruiter_questions_with_claude(resume, job, questions)

                # Validate the answers
                validated_answers = [RecruiterQuestion.model_validate(answer) for answer in answers]

                state["recruiter_answers"] = validated_answers
                new_message = AIMessage(content="✅ Recruiter questions answered with Claude AI + Pydantic validation")
                print("✅ Recruiter questions answered with Claude AI + Pydantic validation")
            else:
                new_message = AIMessage(content="ℹ️ No recruiter questions provided")
                print("ℹ️ No recruiter questions provided")

            state["current_step"] = "recruiter_questions_handled"
            state["messages"] = state.get("messages", []) + [new_message]

        except Exception as e:
            state["errors"].append(f"Recruiter questions handling error: {str(e)}")
            error_message = AIMessage(content=f"❌ Recruiter questions handling failed: {e}")
            state["messages"] = state.get("messages", []) + [error_message]
            print(f"❌ Recruiter questions handling failed: {e}")

        return state

    def finalize_application_node(self, state: LangGraphApplicationState) -> LangGraphApplicationState:
        """Finalize the application process"""
        state["current_step"] = "application_completed"
        final_message = AIMessage(content="🎉 Application processing completed with Claude AI + Pydantic validation!")
        state["messages"] = state.get("messages", []) + [final_message]
        print("🎉 Application processing completed with Claude AI + Pydantic validation!")
        return state

    def _parse_resume_content_with_claude(self, resume_text: str) -> Dict[str, Any]:
        """Parse resume text using Claude AI into structured format"""

        system_prompt = """You are an expert resume parser. Extract and structure resume information into the specified JSON format. 
        Be thorough and accurate. For skills, identify both technical and soft skills, and categorize them appropriately.
        Track skills learned vs skills used in each job. Calculate years of experience and proficiency levels based on context."""

        prompt = f"""
        Parse the following resume and extract all information into a structured JSON format that matches this schema:

        {{
            "personal_info": {{
                "name": "string",
                "email": "string (email format or null)",
                "phone": "string or null",
                "location": "string or null",
                "linkedin_url": "string (URL format or null)",
                "portfolio_url": "string (URL format or null)"
            }},
            "professional_summary": "string",
            "work_experiences": [
                {{
                    "company": "string",
                    "position": "string",
                    "start_date": "string",
                    "end_date": "string",
                    "duration_months": "integer",
                    "location": "string or null",
                    "employment_type": "full_time|part_time|contract|freelance|internship",
                    "responsibilities": ["string", ...],
                    "achievements": ["string", ...],
                    "skills_used": ["string", ...],
                    "skills_learned": ["string", ...],
                    "technologies": ["string", ...],
                }}
            ],
            "education": [
                {{
                    "institution": "string",
                    "degree": "string",
                    "field_of_study": "string",
                    "level": "high_school|associate|bachelor|master|doctorate|certificate",
                    "graduation_date": "string or null",
                    "gpa": "float or null",
                    "relevant_coursework": ["string", ...],
                    "honors": ["string", ...],
                    "activities": ["string", ...]
                }}
            ],
            "skills": [
                {{
                    "name": "string",
                    "category": "technical|soft|domain_specific|language|certification",
                    "proficiency_level": "beginner|intermediate|advanced|expert",
                    "years_experience": "integer or null",
                    "context": "string or null"
                }}
            ],
            "certifications": [
                {{
                    "name": "string",
                    "issuing_organization": "string",
                    "issue_date": "string or null",
                    "expiry_date": "string or null",
                    "credential_id": "string or null"
                }}
            ],
            "projects": [
                {{
                    "name": "string",
                    "description": "string",
                    "technologies": ["string", ...],
                    "role": "string or null",
                    "achievements": ["string", ...]
                }}
            ],
            "languages": [
                {{
                    "name": "string",
                    "proficiency": "native|fluent|conversational|basic"
                }}
            ],
            "volunteer_experience": ["string", ...],
            "publications": ["string", ...],
            "awards": ["string", ...]
        }}

        Resume text:
        {resume_text}

        Return only the JSON object, no additional text.
        """

        response = self._call_claude(prompt, system_prompt)

        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]

            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Claude response: {response}")
            raise ValueError(f"Failed to parse Claude response as JSON: {e}")

    def _parse_job_description_content_with_claude(self, job_desc_text: str) -> Dict[str, Any]:
        """Parse job description using Claude AI into structured format"""

        system_prompt = """You are an expert job description analyzer. Extract and structure job posting information into the specified JSON format.
        Identify requirements vs preferences, categorize skills, and extract company culture information accurately."""

        prompt = f"""
        Parse the following job description and extract all information into a structured JSON format:

        {{
            "company": {{
                "name": "string",
                "size": "string or null",
                "location": "string or null",
                "website": "string (URL format or null)",
                "description": "string or null"
            }},
            "position": "string",
            "location": "string",
            "job_type": "full_time|part_time|contract|freelance|internship",
            "salary_range": "string or null",
            "remote_options": "boolean",
            "summary": "string",
            "responsibilities": ["string", ...],
            "requirements": [
                {{
                    "skill": "string",
                    "importance": "required|preferred|nice_to_have",
                    "category": "technical|soft|domain_specific|language|certification",
                    "years_required": "integer or null",
                    "description": "string or null"
                }}
            ],
            "preferred_qualifications": ["string", ...],
            "benefits": ["string", ...],
            "company_culture": ["string", ...],
            "application_deadline": "string or null"
        }}

        Job description:
        {job_desc_text}

        Return only the JSON object, no additional text.
        """

        response = self._call_claude(prompt, system_prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]

            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Claude response: {response}")
            raise ValueError(f"Failed to parse Claude response as JSON: {e}")

    def _analyze_skill_matching_with_claude(self, resume: ParsedResume, job: ParsedJobDescription) -> Dict[str, Any]:
        """Analyze skill matching using Claude AI"""

        system_prompt = """You are an expert HR analyst specializing in skill matching and candidate assessment.
        Analyze how well a candidate's skills match job requirements. Provide detailed scoring and recommendations."""

        # Prepare resume skills summary
        resume_skills = [
            {
                "name": skill.name,
                "category": skill.category,
                "proficiency": skill.proficiency_level,
                "years": skill.years_experience,
            }
            for skill in resume.skills
        ]

        # Prepare job requirements summary
        job_requirements = [
            {
                "skill": req.skill,
                "importance": req.importance,
                "category": req.category,
                "years_required": req.years_required,
            }
            for req in job.requirements
        ]

        prompt = f"""
        Analyze the skill match between this candidate's resume and job requirements:

        CANDIDATE SKILLS:
        {json.dumps(resume_skills, indent=2)}

        CANDIDATE EXPERIENCE:
        Total Years: {resume.get_total_experience_years()}
        Recent Positions: {[f"{exp.position} at {exp.company}" for exp in resume.work_experiences[:3]]}

        JOB REQUIREMENTS:
        {json.dumps(job_requirements, indent=2)}

        JOB DETAILS:
        Position: {job.position}
        Company: {job.company.name}

        Provide a detailed skill match analysis in this JSON format:

        {{
            "overall_match_score": "float (0-100)",
            "required_skills_match_score": "float (0-100)",
            "matched_requirements": [
                {{
                    "requirement": {{
                        "skill": "string",
                        "importance": "required|preferred|nice_to_have",
                        "category": "technical|soft|domain_specific|language|certification",
                        "years_required": "integer or null"
                    }},
                    "resume_skill": {{
                        "name": "string",
                        "category": "technical|soft|domain_specific|language|certification",
                        "proficiency_level": "beginner|intermediate|advanced|expert",
                        "years_experience": "integer or null"
                    }},
                    "match_strength": "float (0.0-1.0)",
                    "gap_description": "string or null"
                }}
            ],
            "unmatched_requirements": [
                {{
                    "skill": "string",
                    "importance": "required|preferred|nice_to_have",
                    "category": "technical|soft|domain_specific|language|certification",
                    "years_required": "integer or null"
                }}
            ],
            "skill_gaps": [
                {{
                    "skill": "string",
                    "importance": "required|preferred|nice_to_have",
                    "category": "technical|soft|domain_specific|language|certification",
                    "years_required": "integer or null"
                }}
            ],
            "transferable_skills": [
                {{
                    "name": "string",
                    "category": "technical|soft|domain_specific|language|certification",
                    "proficiency_level": "beginner|intermediate|advanced|expert",
                    "years_experience": "integer or null"
                }}
            ],
            "recommendations": ["string", ...]
        }}

        Return only the JSON object, no additional text.
        """

        response = self._call_claude(prompt, system_prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]

            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Claude response: {response}")
            raise ValueError(f"Failed to parse Claude response as JSON: {e}")

    def _generate_cover_letter_with_claude(
        self, resume: ParsedResume, job: ParsedJobDescription, skill_analysis: SkillMatchAnalysis
    ) -> Dict[str, Any]:
        """Generate a tailored cover letter using Claude AI"""

        system_prompt = """You are an expert career counselor and professional writer specializing in cover letters.
        Create compelling, personalized cover letters that highlight relevant experience and demonstrate genuine interest in the role."""

        # Prepare context
        top_skills = [
            match.requirement.skill
            for match in sorted(skill_analysis.matched_requirements, key=lambda x: x.match_strength, reverse=True)[:5]
        ]

        recent_experience = resume.work_experiences[0] if resume.work_experiences else None

        prompt = f"""
        Create a professional cover letter for this job application:

        CANDIDATE INFO:
        Name: {resume.personal_info.name}
        Email: {resume.personal_info.email}
        Phone: {resume.personal_info.phone}
        Location: {resume.personal_info.location}
        
        Total Experience: {resume.get_total_experience_years()} years
        Recent Role: {recent_experience.position if recent_experience else "N/A"} at {recent_experience.company if recent_experience else "N/A"}
        Key Achievements: {recent_experience.achievements[:3] if recent_experience else []}

        JOB DETAILS:
        Position: {job.position}
        Company: {job.company.name}
        Location: {job.location}

        SKILL MATCH:
        Match Score: {skill_analysis.overall_match_score:.1f}%
        Top Matching Skills: {top_skills}
        Key Responsibilities: {job.responsibilities[:3]}

        Generate a cover letter in this JSON format:

        {{
            "header": "string (formatted header with contact info and date)",
            "opening_paragraph": "string (engaging opening that mentions the role and company)",
            "body_paragraphs": [
                "string (experience paragraph highlighting relevant background)",
                "string (skills paragraph showcasing technical abilities)",
                "string (value proposition paragraph explaining unique contributions)"
            ],
            "closing_paragraph": "string (professional closing with call to action)",
            "signature": "string (professional sign-off with name)"
        }}

        Make the cover letter:
        - Professional yet engaging
        - Specific to this role and company
        - Highlighting the strongest skill matches
        - Demonstrating knowledge of the industry/company
        - 300-400 words total
        - ATS-friendly formatting

        Return only the JSON object, no additional text.
        """

        response = self._call_claude(prompt, system_prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]

            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Claude response: {response}")
            raise ValueError(f"Failed to parse Claude response as JSON: {e}")

    def _answer_recruiter_questions_with_claude(
        self, resume: ParsedResume, job: ParsedJobDescription, questions: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate answers to recruiter questions using Claude AI"""

        system_prompt = """You are an expert interview coach and career counselor. 
        Generate thoughtful, professional answers to recruiter questions based on the candidate's background and the specific job opportunity."""

        # Prepare candidate context
        candidate_context = {
            "name": resume.personal_info.name,
            "total_experience": resume.get_total_experience_years(),
            "recent_role": f"{resume.work_experiences[0].position} at {resume.work_experiences[0].company}"
            if resume.work_experiences
            else "N/A",
            "key_skills": [skill.name for skill in resume.skills[:10]],
            "achievements": resume.work_experiences[0].achievements[:3] if resume.work_experiences else [],
            "education": f"{resume.education[0].degree} in {resume.education[0].field_of_study}"
            if resume.education
            else "N/A",
        }

        job_context = {
            "position": job.position,
            "company": job.company.name,
            "responsibilities": job.responsibilities[:3],
        }

        prompt = f"""
        Generate professional answers to recruiter questions based on this candidate profile:

        CANDIDATE PROFILE:
        {json.dumps(candidate_context, indent=2)}

        JOB OPPORTUNITY:
        {json.dumps(job_context, indent=2)}

        RECRUITER QUESTIONS:
        {json.dumps(questions, indent=2)}

        For each question, provide a structured response in this JSON format:

        [
            {{
                "question": "string (the original question)",
                "category": "experience|motivation|skills|salary|availability|general",
                "answer": "string (2-3 sentences, professional and specific)",
                "confidence": "float (0.0-1.0, how confident the answer is)"
            }}
        ]

        Guidelines for answers:
        - Be specific and reference actual experience/skills
        - Show genuine interest in the role/company
        - Be confident but not arrogant
        - Keep answers concise but informative
        - Use the STAR method for experience questions
        - Be honest about salary expectations and availability

        Return only the JSON array, no additional text.
        """

        response = self._call_claude(prompt, system_prompt)

        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            json_str = response[json_start:json_end]

            parsed_data = json.loads(json_str)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Claude response: {response}")
            raise ValueError(f"Failed to parse Claude response as JSON: {e}")

    def run_application_process(
        self, resume_text: str, job_description_text: str, recruiter_questions: Optional[List[str]] = None
    ) -> ApplicationResult:
        """Run the complete application process with Claude AI and return Pydantic result"""

        start_time = datetime.now()

        initial_state = LangGraphApplicationState(
            resume_text=resume_text,
            job_description_text=job_description_text,
            parsed_resume=None,
            parsed_job_description=None,
            skill_match_analysis=None,
            cover_letter=None,
            recruiter_questions=recruiter_questions,
            recruiter_answers=None,
            current_step="starting",
            errors=[],
            messages=[],  # Initialize messages list
        )

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Create and validate the result using Pydantic
        result = ApplicationResult(
            parsed_resume=final_state.get("parsed_resume"),
            parsed_job_description=final_state.get("parsed_job_description"),
            skill_match_analysis=final_state.get("skill_match_analysis"),
            cover_letter=final_state.get("cover_letter"),
            recruiter_answers=final_state.get("recruiter_answers"),
            errors=final_state.get("errors", []),
            status=final_state.get("current_step", "completed"),
            processing_time_seconds=processing_time,
        )

        return result
