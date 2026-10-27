# LLM Engineering Interview Preparation

A comprehensive learning resource containing **10 real-world interview questions** on Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems, with detailed implementations, comments, and discussion prompts.

## 📚 Overview

This repository contains an interactive Jupyter notebook (`Interview_Questions.ipynb`) designed for:
- **Students** preparing for LLM/RAG engineering interviews
- **Developers** learning production-grade LLM patterns
- **Educators** teaching advanced NLP and RAG concepts

## 🎯 Topics Covered

| # | Question | Topic | Difficulty |
|---|----------|-------|------------|
| 1 | FAISS Semantic Search | Vector similarity search with FAISS | Intermediate |
| 2 | Document Chunking & Indexing | Text splitting and embedding management | Intermediate |
| 3 | Retrieval Evaluation Metrics | Precision, Recall, and MRR calculation | Advanced |
| 4 | PII Redaction System | Sensitive data protection with regex & NER | Intermediate |
| 5 | Prompt Template Evaluation | Benchmarking prompt designs with LLM APIs | Intermediate |
| 6 | LLM Logging & Analytics | Structured logging and monitoring | Intermediate |
| 7 | Multi-LLM Router | Async routing across multiple LLM providers | Advanced |
| 8 | Topic-Based Summarization | K-means clustering for document summarization | Advanced |
| 9 | Response Comparison | Semantic similarity and factual overlap metrics | Intermediate |
| 10 | Prompt Injection Detection | Security: pattern matching + ML classifiers | Advanced |

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or VS Code with Jupyter extension

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd interview-preparation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Open the notebook**
   ```bash
   jupyter notebook Interview_Questions.ipynb
   ```
   or use VS Code with Jupyter extension

### API Keys (Optional)
For Questions 5, 6, and 7, you may need API keys:

- **OpenAI**: Set `OPENAI_API_KEY` environment variable
- **Anthropic**: Set `ANTHROPIC_API_KEY` environment variable

Create a `.env` file in this directory:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## 📖 How to Use

### For Students
1. **Read the question** (markdown cell) - understand the problem statement
2. **Try solving it** - pause the notebook and code your solution
3. **Review the implementation** - compare with the provided code
4. **Study the comments** - learn optimization techniques and patterns
5. **Discuss** - think through the discussion prompts for deeper understanding

### For Instructors
- Use this as a teaching resource for LLM engineering courses
- Assignments can involve modifying/extending the solutions
- Discussion prompts serve as excellent interview preparation talking points

### Cell Structure
Each question follows this pattern:
```
📝 Markdown Cell    → Problem description, hints, discussion prompts
🐍 Python Cell(s)   → Implementation with detailed comments
🎯 Example Cell     → Demonstration or test cases
```

## 📁 File Structure

```
interview-preparation/
├── Interview_Questions.ipynb       # Main learning notebook (23 cells)
├── requirements.txt                # Python dependencies
├── vector_index.faiss              # Pre-built FAISS index (auto-generated)
├── README.md                       # This file
└── NOTEBOOK_STRUCTURE.md          # Detailed architecture documentation
```

## 📦 Dependencies

Core libraries used:
- **Embeddings**: `sentence-transformers` - Generate text embeddings
- **Vector Search**: `faiss-cpu` - Fast similarity search
- **Document Processing**: `langchain` - Text splitting utilities
- **NLP**: `spacy` - Named Entity Recognition
- **ML**: `scikit-learn` - Classification and clustering
- **LLM APIs**: `openai`, `anthropic` - Access commercial LLMs
- **Web**: `fastapi` - Build APIs (optional, for Question deployments)

See `requirements.txt` for complete list with versions.

## 🔧 Common Issues & Solutions

### Issue: ImportError for spacy model
```bash
python -m spacy download en_core_web_sm
```

### Issue: FAISS not found
```bash
pip install faiss-cpu
# or for GPU support:
pip install faiss-gpu
```

### Issue: API key errors in Questions 5, 6, 7
- Verify API keys are set in `.env` or environment variables
- Use `python-dotenv` to load `.env` automatically

### Issue: Cell execution fails with "model not found"
First cells download models on first run - this takes time. Wait for completion.

## 💡 Learning Path

**Beginner** → Start with Questions 1-2 (vector search fundamentals)
**Intermediate** → Progress to Questions 3-6 (retrieval and analytics)
**Advanced** → Complete Questions 7-10 (complex systems and security)

## 🎓 Key Concepts Covered

### Vector Embeddings & Search
- Normalized embeddings for cosine similarity
- FAISS index creation and optimization
- Latency optimization for production

### Document Management
- Recursive text chunking with overlap
- Metadata preservation
- Semantic coherence

### Evaluation Metrics
- Precision@K, Recall@K
- Mean Reciprocal Rank (MRR)
- Multi-model benchmarking

### Security
- PII detection with regex and NER
- Prompt injection detection
- Pattern + ML hybrid approaches

### Production Systems
- Structured logging with JSON
- Async multi-LLM routing
- Caching strategies
- Topic clustering and summarization

## 🤝 Contributing

Found improvements? Submit issues or pull requests:
1. Test your changes against all 10 questions
2. Ensure comments are clear for students
3. Verify all imports work

## 📚 Further Reading

- [Langchain Documentation](https://python.langchain.com/)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Anthropic Claude Documentation](https://docs.anthropic.com/)

## 📝 License

This material is provided for educational purposes.

## ❓ Questions or Feedback?

For questions about specific implementations, refer to the **Discussion** prompts in each question cell.

---

**Last Updated**: December 2025  
**Total Questions**: 10  
**Estimated Study Time**: 8-12 hours  
**Notebook Cells**: 23 (organized for optimal learning flow)
