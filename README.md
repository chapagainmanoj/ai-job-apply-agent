# JobCopilot 🚀

AI-powered job application assistant that generates tailored cover letters, answers recruiter questions, and analyzes skill matches using Claude AI.

## Quick Start

### Prerequisites
- Python 3.8+
- [UV package manager](https://docs.astral.sh/uv/)
- Anthropic API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd jobcopilot
   ```

2. **Install dependencies with UV:**
   ```bash
   uv sync
   ```

3. **Setup environment variables:**
   ```bash
   # Create .env file
   echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" > .env
   ```

### Usage

**Run the application:**
```bash
uv run python main.py
```

The system will:
- ✅ Parse your resume
- ✅ Analyze job requirements  
- ✅ Generate tailored cover letter
- ✅ Answer recruiter questions
- ✅ Provide skill match analysis

## Configuration

Edit the sample data in `main.py`:
- Replace `sample_resume` with your resume text
- Replace `sample_job_description` with target job posting
- Update `sample_questions` with recruiter questions
- Feel free to contribute other ways to provide the information

## Features

- 🎯 **Smart Skill Matching** - AI-powered skill analysis and gap identification
- 📝 **Custom Cover Letters** - Tailored to specific job requirements
- ❓ **Recruiter Q&A** - Intelligent answers to application questions  
- 📊 **Match Scoring** - Quantified job fit analysis
- 🔄 **LangGraph Workflow** - Multi-step AI processing pipeline

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ANTHROPIC_API_KEY` | Your Anthropic Claude API key | ✅ Yes |

## API Key Setup

Get your API key from [Anthropic Console](https://console.anthropic.com/) and add it to your `.env` file.

## Project Structure

```
jobcopilot/
├── main.py           # Main application entry point
├── agent.py          # Core AI agent system
├── .env             # Environment variables
└── README.md        # This file
```

---

## TODO & Roadmap 📋
- [ ] **Web UI Interface** - Build FastAPI/Streamlit web interface
- [ ] **PDF Resume Upload** - Support PDF resume parsing (PyPDF2/pdfplumber)
- [ ] **Multiple Job Applications** - Batch process multiple job postings
- [ ] **Resume Optimization** - AI-powered resume improvement suggestions
- [ ] **ATS Score Analysis** - Applicant Tracking System compatibility scoring
- [ ] **Database Integration** - SQLite/PostgreSQL for storing applications
- [ ] **Resume Templates** - Multiple professional resume templates
- [ ] **Application History** - Track and manage past applications
- [ ] **LinkedIn Integration** - Auto-apply through LinkedIn API
- [ ] **LinkedIn and Other Job Board APIs** - Auto-apply through API integration - LinkedIn  , Indeed, Glassdoor, Wellfound


**Built with:** Claude AI • LangGraph • Python • UV