# AI Research Assistant  
### *An Intelligent Multi-Agent System for Automated IEEE Research Paper Generation*

> From **research idea → literature → experiment design → IEEE paper → Overleaf-ready ZIP**, all in one click.

---

## 🌟 Overview

The **AI Research Paper Assistant** is a **multi-agent artificial intelligence system** that automates the complete academic research workflow.  
It retrieves real research papers, summarizes them using LLMs, evaluates research quality, designs experiments, and finally generates **IEEE-formatted research papers** with a modern **web interface**.

This project bridges the gap between **academic research** and **AI automation**, reducing weeks of work into minutes.

---

## Key Capabilities

✅ Automated **Research Paper Retrieval**   
✅ **AI-Based Summarization & Gap Detection** 
✅ **Research Quality Evaluation**  
✅ **Experiment Design Automation**  
✅ **IEEE Research Paper Generation**  
✅ **Overleaf-Ready ZIP Export**  
✅ **Download as Markdown & LaTeX**  

---

## System Architecture

User → Streamlit UI
↓
FastAPI Backend
↓
Retrieval Agent → Summarizer Agent → Evaluator Agent
↓
Designer Agent → Report Writer Agent
↓
Markdown + IEEE LaTeX + Overleaf ZIP

---


Each agent performs a **dedicated cognitive task**, closely mimicking how a real research team operates.

---

## Agents in the System

| Agent Name | Responsibility |
|-----------|----------------|
| Retrieval Agent | Fetches latest research papers |
| Summarizer Agent | Generates structured summaries |
| Evaluator Agent | Validates research quality |
| Designer Agent | Designs experiments |
| Report Writer Agent | Generates IEEE papers |
| Memory Store | Persistent research memory |
| History Manager | Stores past reports |

---

## Technology Stack

- **Backend:** FastAPI, Python
- **Frontend:** Streamlit
- **APIs:** Semantic Scholar, Google CSE
- **Formats:** Markdown, IEEE LaTeX

---

## User Workflow 

**Enter Research Topic** :
The user inputs a research topic in the Streamlit interface.

**Run the AI Research Pipeline** :
On clicking “Run Research”, the complete multi-agent pipeline is triggered.

**Automated Multi-Agent Execution** : 
The system sequentially activates:

**Retrieval Agent** – Fetches real-time research papers from Semantic Scholar

**Summarizer Agent** – Produces structured summaries and extracts key insights

**Evaluator Agent** – Validates research quality and relevance

**Designer Agent** – Generates an experimental design

**Report Writer Agent** – Creates:

    A Markdown research report

    An IEEE-compliant LaTeX paper

**Preview & Download** : The user can:

    Preview the paper inside the UI

    Download the Markdown file

    Download the Overleaf-ready ZIP (LaTeX + Bib + IEEE class)

**Persistent History** : 
All generated reports are saved and displayed under Previous Reports for future access.

<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/8dabc21b-52d8-4944-895c-9a94a074d7bb"
    width="200"
  />
</p>

## How to Clone & Run the Project

1️⃣ Clone the Repository

    git clone https://github.com/Kritik2310/AI-Research-Paper-Generator.git

    cd AI-Research-Paper-Generator

2️⃣ Create Virtual Environment

    python -m venv .venv
    .venv\Scripts\activate

3️⃣ Install Dependencies

    pip install -r requirements.txt

4️⃣ Configure Environment Variables

Create a .env file in the root directory:

    SEMANTIC_SCHOLAR_API_KEY=your_key
    GEMINI_API_KEY=your_key
    GOOGLE_API_KEY=your_key
    GOOGLE_CSE_ID=your_key

5️⃣ Run Backend (FastAPI)

    uvicorn pipeline:app --reload

6️⃣ Run Streamlit UI

    streamlit run ui.py

----
