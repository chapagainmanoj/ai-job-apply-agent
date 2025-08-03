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

from utils import extract_dict_from_json_response


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
        self.model = init_chat_model(model, api_key=api_key, timeout=200)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(LangGraphApplicationState)

        workflow.add_node("parse_resume", self.parse_resume_node)
        workflow.add_node("parse_job_description", self.parse_job_description_node)
        workflow.add_node("analyze_skill_match", self.analyze_skill_match_node)
        workflow.add_node("generate_cover_letter", self.generate_cover_letter_node)
        workflow.add_node("handle_recruiter_questions", self.handle_recruiter_questions_node)
        workflow.add_node("finalize_application", self.finalize_application_node)

        workflow.set_entry_point("parse_resume")

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
            core = self._parse_resume_content_with_claude(resume_text)
            skills = self._parse_resume_skills_with_claude(resume_text)
            parsed_resume = {**core, "skills": skills}

            validated_resume = ParsedResume.model_validate(parsed_resume)

            state["parsed_resume"] = validated_resume
            state["current_step"] = "resume_parsed"

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
        Be thorough and accurate.
        Track skills learned vs skills used in each job.

        Extract and structure the following sections into JSON:
        - personal_info
        - professional_summary
        - work_experiences
        - education
        - certifications
        - projects
        - languages
        - volunteer_experience
        - publications
        - awards
        
        IMPORTANT: Return ONLY valid JSON. No explanations, no markdown formatting, no additional text."""

        prompt = f"""
        Parse the following resume and extract all information into a structured JSON format that matches this schema:

        {{
            "personal_info": {{
                "name": "string",
                "email": "string (email format or null)",
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
        return extract_dict_from_json_response(response)

    def _parse_resume_skills_with_claude(self, resume_text: str) -> List[Dict[str, Any]]:
        """
        Parse only the skills section(s) of a resume into structured JSON via Claude.
        Identifies skills learned vs used, categorizes, infers proficiency and years.
        """
        system_prompt = """
            You are an expert skills extractor. From the resume text, identify relevant skill mentioned
            and output a JSON array of skill objects.
            For now only extract maximum of 15 skills based on context and relevancy.
            For skills, identify both technical and soft skills, and categorize them appropriately.
            Calculate years of experience and proficiency levels based on context.
            For each skill, include:
            - name
            - category (technical|soft|domain_specific|language|certification)
            - proficiency_level (beginner|intermediate|advanced|expert)
            - years_experience (integer or null)
            - context (brief phrase: e.g. “used in X project” or “learned at Y”)

            IMPORTANT: Return ONLY valid JSON array. No extra text.
            """

        prompt = f"""
            Extract all skills from this resume text and structure them as:

            [
            {{
                "name": "string",
                "category": "technical|soft|domain_specific|language|certification",
                "proficiency_level": "beginner|intermediate|advanced|expert",
                "years_experience": integer or null,
                "context": "string or null"
            }},
            ...
            ]

            Resume text:
            {resume_text}

            Return only the JSON array.
            """

        response = self._call_claude(prompt, system_prompt)
        return extract_dict_from_json_response(response, "list")

    def _parse_job_description_content_with_claude(self, job_desc_text: str) -> Dict[str, Any]:
        """Parse job description using Claude AI into structured format"""

        system_prompt = """You are an expert job description analyzer. Extract and structure job posting information into the specified JSON format.
        Identify requirements vs preferences, categorize skills, and extract company culture information accurately.

        IMPORTANT: Return ONLY valid JSON. No explanations, no markdown formatting, no additional text."""

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

        return extract_dict_from_json_response(response)

    def _analyze_skill_matching_with_claude(self, resume: ParsedResume, job: ParsedJobDescription) -> Dict[str, Any]:
        """Analyze skill matching using Claude AI"""

        system_prompt = """You are an expert HR analyst specializing in skill matching and candidate assessment.
        Analyze how well a candidate's skills match job requirements. Provide detailed scoring and recommendations."""

        resume_skills = [
            {
                "name": skill.name,
                "category": skill.category,
                "proficiency": skill.proficiency_level,
                "years": skill.years_experience,
            }
            for skill in resume.skills
        ]

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

        return extract_dict_from_json_response(response)

    def _generate_cover_letter_with_claude(
        self, resume: ParsedResume, job: ParsedJobDescription, skill_analysis: SkillMatchAnalysis
    ) -> Dict[str, Any]:
        """Generate a tailored cover letter using Claude AI"""

        system_prompt = """
                You are a senior career strategist and award-winning copywriter for tech roles.
                Your mission: craft a cover letter that
                1. Hooks the reader by naming a product/mission insight.
                2. Delivers a TL;DR with top achievements.
                3. Weaves a short narrative showing impact (with metrics).
                4. Lists core skills tied to role requirements.
                5. Connects personal values to company culture.
                6. Closes with a confident call to action.
                Ensure it scans well for ATS but reads naturally.
                """

        top_skills = [
            match.requirement.skill
            for match in sorted(skill_analysis.matched_requirements, key=lambda x: x.match_strength, reverse=True)[:5]
        ]

        recent = resume.work_experiences[0] if resume.work_experiences else None

        prompt = f"""
        Use the SYSTEM instructions above and output EXACTLY this JSON:

        {{
            "header": "{resume.personal_info.name} | {resume.personal_info.email} | {resume.personal_info.location}\\n{date.today().isoformat()}",
            "tldr": "• {resume.get_total_experience_years()} yrs experience • Top skills: {", ".join(top_skills)} • Recent: {recent.position if recent else "N/A"} at {recent.company if recent else "N/A"}",
            "opening": "2-3 sentences. Start with a specific compliment or insight about {job.company.name}. Mention the {job.position} role by name.",
            "story_paragraph": "3-4 sentences. Describe a past project where you delivered X (metric) that maps directly to a core responsibility: {job.responsibilities[0]}.",
            "skills_paragraph": "3-4 sentences. Call out your top 3–5 skills ({", ".join(top_skills)}) and how each will solve a key challenge for {job.company.name}. Include one numeric result per skill.",
            "culture_fit": "2-3 sentences. Explain why {job.company.name}’s mission or values resonate with your career goals.",
            "closing": "2 sentences. Express enthusiasm, request next steps, and thank the reader.",
            "signature": "Best regards, {resume.personal_info.name}"
        }}

        Length: ~350 words total. No extra keys or commentary—only the JSON above.
        """

        response = self._call_claude(prompt, system_prompt)

        return extract_dict_from_json_response(response)

    def _answer_recruiter_questions_with_claude(
        self, resume: ParsedResume, job: ParsedJobDescription, questions: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate answers to recruiter questions using Claude AI"""

        system_prompt = """You are an expert interview coach and career counselor. 
        Generate thoughtful, professional answers to recruiter questions based on the candidate's background and the specific job opportunity."""

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

        return extract_dict_from_json_response(response, type="list")

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
