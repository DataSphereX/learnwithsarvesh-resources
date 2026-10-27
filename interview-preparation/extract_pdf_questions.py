import pdfplumber
import re

pdf_path = r'c:\Users\sarve\Downloads\GEN AI CODING SOLUTION - Colab.pdf'

with pdfplumber.open(pdf_path) as pdf:
    full_text = ''
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + '\n'

# Find all numbered questions (1., 2., 3., etc.)
lines = full_text.split('\n')
questions = []

for line in lines:
    # Look for pattern like "1. ", "2. ", etc at start of line
    match = re.match(r'^(\d+)\.\s+(.+)', line.strip())
    if match:
        q_num = match.group(1)
        q_text = match.group(2)
        questions.append((q_num, q_text))

print(f"Total questions found: {len(questions)}\n")
print("=== ALL QUESTIONS IN PDF ===\n")

for idx, (num, text) in enumerate(questions, 1):
    # Truncate long text
    if len(text) > 100:
        text = text[:100] + "..."
    print(f"{idx}. Q{num}: {text}")
