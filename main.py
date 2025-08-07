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
    Manoj Chapagain
    chapagainmanoj35@gmail.com | manoj.codes | linkedin.com/in/chapagainmanoj | github.com/chapagainmanoj
    Summary Statement
    Experienced software engineer with 8 years in web development, distributed systems, and cybersecurity. Proven
    track record building and leading teams to deliver high-quality products on tight deadlines across fintech,
    cybersecurity, IoT, and SaaS sectors.
    Experience
    Full-Stack Developer Feb. 2024 – Present
    Mechatronic Canada
    • Led end-to-end development across frontend, backend, CI/CD pipelines, and production infrastructure using
    DevOps practices
    • Developed a firmware module for IoT devices and built the user-facing web interface
    • Developed AI agents, tools, and automations for efficient and scalable workflows
    Backend Developer May. 2022 – Apr. 2023
    Microsec Singapore
    • Developed lightweight PKI-based security protocols in Python for IoT/OT devices using the Cryptography library;
    enabled certificate provisioning, encrypted messaging, and integrity checks for resource-constrained environments
    • Architected and deployed microservices with Docker and Kubernetes, orchestrating services across a multi-node
    cluster to ensure high availability and scalability
    Sr. Software Engineer Jan. 2021 – Mar. 2022
    Codavatar Kathmandu, Nepal
    • Led the backend team in delivering scalable, production-ready services under tight deadlines, ensuring on-time
    delivery of key milestones
    • Developed and optimized core features for Krispcall, a cloud-based phone system leveraging WebRTC and Twilio
    • Adopted domain-driven and clean architecture principles, applying design patterns to enhance code quality and
    maintainability
    • Performed performance tuning and refactoring to improve system scalability, reliability, and response times
    Full Stack Developer Nov. 2017 – Dec. 2020
    Janaki Tech. Lalitpur, Nepal
    • Built the front-end from scratch using React, growing the user base from few thousand to 100K within one year for
    an early-stage fintech startup
    • Engineered a payment gateway and digital wallet using Python, Django, React.js, and AWS; integrated with banks,
    telecoms, and e-commerce platforms via third-party APIs
    Jr. Full-Stack Developer Dec. 2016 – Nov. 2017
    Grepsr Kathmandu, Nepal
    • Built and maintained web scraping and data-extraction pipelines using microservice architecture, gRPC, Docker,
    and Kubernetes
    Education
    Project Managemnt - IT May. 2023 – Dec. 2023
    Seneca College Toronto, Canada
    Bachelor’s degree in Computer Engineering Nov. 2012 – Sep. 2016
    Tribhuvan University, IOE Pulchowk Campus Lalitpur, Nepal
    Open Source Projects
    Employee Management System | Python, Django, drf, React, PostgreSQL, Docker | Source June 2020
    databases-extensions | SqlAlchemy, Postgres, GraphQL, Cursor Pagination| Source May 2021 – Present
    Publications
    Effects of Auto-Tuning and Pitch Normalization on Query by Humming June 2020
    International Journal of Advanced engineering Source
    Technical Skills
    Languages/Scripts: Python, JavaScript/TypeScript, SQL, Shell (sh, zsh, bash)
    Frameworks/Libraries: FastAPI, Flask, Django, DRF, Starlette, gRPC, GraphQL (Ariadne, Graphene),
    SQLAlchemy, RabbitMQ, Redis, OpenGL, Pandas, Matplotlib, React, Node.js, Selenium, LangGraph, OpenAI API,
    Pinecone, scikit-learn
    DevOps/Cloud: Docker, Kubernetes (K8s), Ansible, Nginx, CI/CD (GitHub Actions, GitLab CI), AWS (EC2, S3,
    RDS, Lambda), Terraform, Prometheus, Grafana
    Testing/Docs: Pytest, Postman, Swagger/OpenAPI, MkDocs, Ruff, Flake8, Cypress
    Others: Linux, Jira, Asana
    """

    sample_job_description = """
    Your mission
    You'll play a key role in the success of deepset AI Platform, our SaaS platform for Large Language Model (LLM) applications, empowering users to build their own LLM-powered features.

    You'll join our cross-functional Product team, and work closely with Product, Design and other engineers.

    You'll contribute to building a cutting-edge AI platform capable of handling high data volumes, enabling users to deploy scalable AI applications that manage large-scale traffic efficiently.

    Our mission
    We are an AI startup that empowers developers to build applications that use natural language as the interface to data. Our open-source framework Haystack is used by thousands of developers to build LLM applications. deepset AI Platform (formerly known as deepset Cloud) enables enterprises worldwide to unlock the potential of large language models for their business cases. Come join our team and work with tech experts on real-world cases, in a fast-paced and high-impact environment, and shape the AI space together with us.

    Life at deepset
    At deepset you will find an environment with equal opportunities to grow, share ideas, build relationships, and succeed. We believe every employee is important and generates a direct impact on our team’s success.

    We are certified by Flexa® Careers as a flexible employer, and our remote first culture is designed to provide work-life balance, autonomy, and flexibility. Our daily meetings, the in person "re:base", and the social events allow us to work with great synergy. We would like to invite you to visit our life page to learn more about our culture.

    Your impact
    Design and build an AI platform that empowers customers to solve real business challenges using state-of-the-art technology and customer-driven insights.
    Develop scalable infrastructure to support thousands of LLM applications processing millions of documents, with the ability to handle 100s of user requests per second on our SaaS platform powered by Haystack, our open-source library.
    Own features end-to-end, from ideation to production. This includes product and architecture design, CI/CD pipelines, implementation & testing, production monitoring, and incorporating customer feedback.
    Collaborate with our Open Source team to identify opportunities for improving Haystack and contribute enhancements to Haystack.
    Engage with infrastructure topics and collaborate with the infrastructure team for support and debugging assistance.
    Take on "firefighter" duties, addressing urgent issues with direct customer exposure alongside the Solutions Engineering team. Maintain and enhance debugging processes to ensure system reliability.
    Mentor other engineers and help them grow in their careers.
    Your profile
    5+ years of experience in backend development with proficiency in Python, ideally with experience in early-stage product development or working in growth/startup environments.
    Expertise in building and managing REST APIs, SQL databases, CI/CD pipelines, Kubernetes, and cloud services. Familiarity with Infrastructure-as-Code tools (e.g. Terraform or Crossplane) is a plus.
    A strong commitment to delivering well-tested, maintainable, and observable code.
    Strong problem-solving skills with the ability to quickly troubleshoot and prioritize issues, especially in high-pressure situations or when working with code from other teams.
    Confidence in managing customer-facing challenges, with the ability to handle critical incidents effectively.
    Proactive and results-oriented, with the ability to contribute constructively and align with team decisions.

    About us
    Our vision: We make machines understand language so that humans can achieve more.

    ‍
    ‍Our mission: To make custom AI solutions accessible to every organization, driving adoption and impact. By combining innovation with expertise, we simplify the complexity of LLM agent and application development, empowering teams to solve their most mission-critical challenges with speed, trust, and control.
    """

    sample_questions = [
        "Do you have previous experience with open source? If so, please share more details:*",
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
    print("\n📊 PYDANTIC MODEL RESULTS:")
    if result.cover_letter:
        print("\n📝 COVER LETTER:")
        print(result.cover_letter.get_full_letter())

    if result.recruiter_answers:
        print("\n❓ RECRUITER ANSWERS:")
        for answer in result.recruiter_answers:
            print(f"  Q: {answer.question}")
            print(f"  A: {answer.answer}")
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
