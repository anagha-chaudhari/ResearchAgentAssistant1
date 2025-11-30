---
title: Research Assistant Backend API
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🔬 Research Assistant Backend API

FastAPI multi-agent system for automated research paper analysis and IEEE LaTeX generation.

## 🌐 API Endpoints

- `POST /run_pipeline` - Complete research workflow
- `POST /download-zip` - Download Overleaf package
- `GET /progress/{session_id}` - Track progress
- `GET /docs` - API documentation

## 🔐 Required Secrets

Add in Space Settings:
- SEMANTIC_SCHOLAR_API_KEY
- GEMINI_API_KEY
- GOOGLE_API_KEY
- GOOGLE_CSE_ID

## 🧰 Tech Stack

FastAPI | Google Gemini | Semantic Scholar | Docker
