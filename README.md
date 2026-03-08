py -3.11 -m venv .venv
.\.venv\Scripts\activate

# Lesson 02
cd ".\lessons\Lesson02-langchain-basics"
py -m pip install -r requirements.txt
py pdf_summarizer.py

# Lesson 03 — from project root with venv activated
.\.venv\Scripts\activate
py -m pip install -r lessons\lesson03-agent\requirements.txt
py lessons\lesson03-agent\sandbox.py