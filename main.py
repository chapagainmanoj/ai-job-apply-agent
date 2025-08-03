import os
from agent import ResumeJobApplicationSystem
from dotenv import load_dotenv

load_dotenv()


# Usage example with Pydantic validation and LangGraph + Anthropic
def main():
    """Example usage of the LangGraph + Claude-powered system"""

    # Set the API key as environment variable
    # Make sure you have ANTHROPIC_API_KEY in your .env file
    API_KEY = os.getenv("ANTHROPIC_API_KEY")

    if not API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")

    system = ResumeJobApplicationSystem(
        api_key=API_KEY,
        model="anthropic:claude-3-5-sonnet-latest",  # Updated to use LangChain format
    )

    # Sample data (same as before)
    sample_resume = """
    John Doe
    Software Engineer
    john.doe@email.com | (555) 123-4567
    
    PROFESSIONAL SUMMARY
    Experienced software engineer with 5+ years developing web applications using Python and JavaScript.
    
    EXPERIENCE
    
    Senior Software Engineer - TechCorp Inc
    Jan 2020 - Present
    • Led development of microservices architecture using Python and AWS
    • Implemented CI/CD pipelines with Jenkins and Docker
    • Mentored junior developers and improved code quality
    • Reduced system latency by 40% through optimization
    
    Software Developer - StartupXYZ
    Jun 2018 - Dec 2019
    • Developed full-stack web applications using React and Node.js
    • Integrated third-party APIs and payment systems
    • Collaborated with cross-functional teams in Agile environment
    • Implemented automated testing reducing bugs by 30%
    
    EDUCATION
    Bachelor of Science in Computer Science
    University of Technology - 2018
    
    SKILLS
    Programming Languages: Python, JavaScript, Java
    Frameworks: React, Node.js, Django
    Cloud: AWS, Docker, Kubernetes
    Tools: Git, Jenkins, JIRA
    """

    sample_job_description = """
    Senior Full Stack Developer
    InnovateTech Solutions
    
    We are seeking a Senior Full Stack Developer to join our growing team.
    
    RESPONSIBILITIES:
    • Design and develop scalable web applications
    • Work with modern JavaScript frameworks and Python backend
    • Collaborate with product managers and designers
    • Mentor junior developers
    • Implement best practices for code quality and testing
    
    REQUIREMENTS:
    • 3+ years experience with Python and JavaScript
    • Experience with React or similar frontend frameworks
    • Knowledge of cloud platforms (AWS preferred)
    • Strong problem-solving and communication skills
    • Experience with Agile development methodologies
    
    PREFERRED:
    • Experience with microservices architecture
    • DevOps experience with Docker/Kubernetes
    • Leadership experience
    
    We offer competitive salary, health benefits, and remote work options.
    """

    sample_questions = [
        "Why are you interested in this position?",
        "What is your experience with microservices?",
        "How do you handle tight deadlines?",
        "What are your salary expectations?",
    ]

    # Run the application process
    print("🚀 Starting LangGraph + Claude-powered application process...")
    result = system.run_application_process(
        resume_text=sample_resume, job_description_text=sample_job_description, recruiter_questions=sample_questions
    )

    # Display results using Pydantic methods
    print("\n📊 PYDANTIC VALIDATED RESULTS:")
    print(f"Status: {result.status}")
    print(f"Processing Time: {result.processing_time_seconds:.2f} seconds")

    if result.errors:
        print(f"\n❌ Errors: {result.errors}")

    # Use Pydantic model methods
    summary = result.get_summary()
    print("\n📈 SUMMARY:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if result.skill_match_analysis:
        print("\n🎯 SKILL ANALYSIS:")
        print(f"  Overall Match: {result.skill_match_analysis.overall_match_score:.1f}%")
        print(f"  Required Skills: {result.skill_match_analysis.required_skills_match_score:.1f}%")
        print(f"  Good Match: {result.skill_match_analysis.is_good_match()}")

    if result.cover_letter:
        print("\n📝 COVER LETTER:")
        print(f"  Word Count: {result.cover_letter.get_word_count()}")
        print(f"  First 100 chars: {result.cover_letter.get_full_letter()[:100]}...")

    if result.recruiter_answers:
        print("\n❓ RECRUITER ANSWERS:")
        for answer in result.recruiter_answers:
            print(f"  Q: {answer.question}")
            print(f"  Category: {answer.category}")
            print(f"  Confidence: {answer.confidence:.2f}")
            print(f"  High Confidence: {answer.is_high_confidence()}")
            print()

    # Export results to JSON using Pydantic
    print("\n💾 JSON Export Available:")
    json_output = result.export_to_json()
    print(f"  JSON Length: {len(json_output)} characters")

    return result


if __name__ == "__main__":
    main()
