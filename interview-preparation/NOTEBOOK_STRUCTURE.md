# Interview Questions Notebook - Complete Structure

## 📚 Overview

This Jupyter notebook contains **10 professionally-designed interview questions** focused on **LLM Engineering & RAG (Retrieval-Augmented Generation)** systems.

**Total Cells**: 23 cells (10 markdown question headers + 10 code solution cells + 1 demo + 1 summary + 1 blank)

---

## 🎯 Complete Question Breakdown

### **Question 1: FAISS Semantic Search** (Cells 1-3)
- **Topic**: Fast vector similarity search
- **Skills**: Embeddings, FAISS indexing, normalization
- **Key Concepts**:
  - Normalized embeddings for cosine similarity
  - L2 distance in FAISS
  - Float32 optimization
  - Sub-200ms latency requirement
- **Output**: Returns top-3 semantic matches with distances

---

### **Question 2: Document Chunking & Indexing** (Cells 4-5)
- **Topic**: Preparing documents for RAG pipelines
- **Skills**: RecursiveCharacterTextSplitter, metadata management
- **Key Concepts**:
  - Chunk size selection (500 chars)
  - Overlap management (50 chars)
  - Metadata enrichment (title, source, timestamp)
  - Embedding generation for chunks
- **Output**: List of indexed chunks with metadata

---

### **Question 3: Retrieval Evaluation Metrics** (Cells 6-7)
- **Topic**: Measuring retrieval system quality
- **Skills**: Precision@k, Mean Reciprocal Rank (MRR), metrics design
- **Key Concepts**:
  - Precision@5 calculation
  - MRR (reciprocal rank) for first relevant result
  - Use case-specific metric selection
  - Vectorized numpy operations
- **Output**: Comparison table of model performance

---

### **Question 4: PII Redaction System** (Cells 8-9)
- **Topic**: Privacy & compliance in LLM applications
- **Skills**: Regex patterns, Named Entity Recognition (NER), GDPR compliance
- **Key Concepts**:
  - Two-layer hybrid approach (regex + NER)
  - Email, phone, SSN, credit card pattern matching
  - spaCy entity recognition (PERSON, ORG, GPE, LOC)
  - Privacy-critical redaction thresholds
- **Output**: Sanitized queries with redaction summary

---

### **Question 5: Prompt Template Evaluation** (Cells 10-11)
- **Topic**: Optimizing prompt engineering
- **Skills**: Template design, A/B testing, metrics definition
- **Key Concepts**:
  - JSON-based prompt templates
  - Accuracy evaluation (keyword matching)
  - Fluency scoring
  - Template comparison framework
- **Output**: Performance comparison and best template selection

---

### **Question 6: LLM Logging & Analytics** (Cells 12-13)
- **Topic**: Production monitoring and feedback loops
- **Skills**: Structured logging, JSONL format, privacy-preserving hashing
- **Key Concepts**:
  - JSON-LD structured logs
  - User ID hashing (SHA-256)
  - Performance metrics (latency, token usage)
  - Feedback integration
  - Data ingestion to Elastic/BigQuery
- **Output**: JSONL log file with user interactions

---

### **Question 7: Multi-LLM Router** (Cells 14-15)
- **Topic**: Cost and latency optimization
- **Skills**: Async/await, caching, routing logic, LRU cache
- **Key Concepts**:
  - Model registry with cost/latency profiles
  - Priority-based routing (speed, quality, cost)
  - Query caching with MD5 hashing
  - Asynchronous parallel calls
  - Cache statistics
- **Output**: Routed response with timing and cost metrics

---

### **Question 8: Topic-Based Summarization** (Cells 16-17)
- **Topic**: Extractive summarization using clustering
- **Skills**: K-Means clustering, sentence embeddings, extractive summary
- **Key Concepts**:
  - Sentence tokenization with NLTK
  - Embedding-based similarity
  - Cluster centroid calculation
  - Representative sentence selection
  - Topic preservation
- **Output**: Multi-topic summary (3 representative sentences)

---

### **Question 9: Response Comparison** (Cells 18-19)
- **Topic**: Evaluating LLM response quality
- **Skills**: Semantic similarity, factual consistency, NER-based evaluation
- **Key Concepts**:
  - Cosine similarity for semantic meaning
  - Jaccard similarity for entity overlap
  - Named entity extraction
  - Multi-dimensional evaluation
  - Verdict generation
- **Output**: Comparison scores with consistency verdict

---

### **Question 10: Prompt Injection Detection** (Cells 20-21)
- **Topic**: Security & attack prevention
- **Skills**: Regex patterns, ML classification, hybrid detection
- **Key Concepts**:
  - Known attack pattern matching
  - Embedding-based semantic detection
  - Logistic regression classifier
  - Confidence thresholding (70%)
  - Two-layer defense system
- **Output**: Safety verdict with confidence score

---

## 📊 Cell Structure Map

```
Cell  Type    Content
----  -----   -------
1     📝      Question 1 Header
2     🐍      Question 1 Solution (FAISS code)
3     🐍      Question 1 Demo (semantic_search call)
4     📝      Question 2 Header
5     🐍      Question 2 Solution (Document chunking)
6     📝      Question 3 Header
7     🐍      Question 3 Solution (Retrieval metrics)
8     📝      Question 4 Header
9     🐍      Question 4 Solution (PII redaction)
10    📝      Question 5 Header
11    🐍      Question 5 Solution (Prompt templates)
12    📝      Question 6 Header
13    🐍      Question 6 Solution (LLM logging)
14    📝      Question 7 Header
15    🐍      Question 7 Solution (Multi-LLM router)
16    📝      Question 8 Header
17    🐍      Question 8 Solution (Topic summarization)
18    📝      Question 9 Header
19    🐍      Question 9 Solution (Response comparison)
20    📝      Question 10 Header
21    🐍      Question 10 Solution (Injection detection)
22    🐍      Blank cell for user notes
23    📝      Summary & Prerequisites
```

---

## 🎓 Learning Path

### Beginner Track
1. Start with **Question 1** (FAISS basics)
2. Move to **Question 2** (Document preparation)
3. Learn **Question 3** (Evaluation metrics)

### Intermediate Track
4. Study **Question 4** (Privacy & compliance)
5. Practice **Question 5** (Prompt engineering)
6. Explore **Question 6** (Production logging)

### Advanced Track
7. Design with **Question 7** (System architecture)
8. Evaluate with **Question 8** (Summarization)
9. Compare with **Question 9** (Quality metrics)
10. Secure with **Question 10** (Attack detection)

---

## 💡 Key Technical Concepts

| Concept | Questions | Priority |
|---------|-----------|----------|
| Embeddings & Vectors | 1, 8, 9 | ⭐⭐⭐ |
| RAG Pipeline | 2, 3, 6 | ⭐⭐⭐ |
| Evaluation Metrics | 3, 5, 9 | ⭐⭐⭐ |
| Security & Privacy | 4, 10 | ⭐⭐⭐ |
| Performance Optimization | 1, 7 | ⭐⭐ |
| Async Programming | 7 | ⭐⭐ |
| ML Classifiers | 10 | ⭐⭐ |

---

## 🔧 Dependencies

```bash
# Core packages
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu
pip install langchain

# NLP & ML
pip install spacy
pip install scikit-learn
pip install nltk

# LLM APIs
pip install openai
pip install anthropic

# API Framework (for Q4)
pip install fastapi
pip install pydantic
```

### Download NLP Models
```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt')"
```

---

## 📝 Code Style & Comments

Each solution includes:

✅ **Section Headers** (clear problem definition)
✅ **Inline Comments** (explaining each step)
✅ **Type Hints** (parameter and return types)
✅ **Docstrings** (function documentation)
✅ **Example Usage** (demonstrate functionality)
✅ **Output Visualization** (print statements with emojis)

---

## 🎯 Interview Preparation Tips

### Before Each Question:
1. Read the question and hint carefully
2. Think about edge cases
3. Consider production implications

### While Implementing:
1. Follow the comment structure provided
2. Understand each optimization mentioned
3. Think about the discussion question

### After Each Question:
1. Run the demo code
2. Experiment with different inputs
3. Answer the discussion question

---

## ✨ Quality Checklist

- ✅ All 10 questions properly numbered (1-10)
- ✅ Consistent markdown formatting with 🎯 emoji
- ✅ Each question has problem description
- ✅ Each question has hint provided
- ✅ Each question has discussion talking point
- ✅ All code cells have detailed comments
- ✅ Type hints on all functions
- ✅ Example usage demonstrated
- ✅ No duplicate questions
- ✅ Logical learning progression

---

## 🚀 Quick Start

1. **Clone/Open** the notebook
2. **Install** dependencies: `pip install -r requirements.txt`
3. **Run cells sequentially** from Question 1 onwards
4. **Modify examples** with your own use cases
5. **Answer discussion** questions for interviews

---

## 📚 Additional Resources

- [FAISS Documentation](https://faiss.ai/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain Documentation](https://python.langchain.com/)
- [spaCy NLP](https://spacy.io/)
- [OpenAI API](https://platform.openai.com/)

---

**Last Updated**: December 20, 2025  
**Format**: Jupyter Notebook (.ipynb)  
**Language**: Python 3.8+
