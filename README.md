# 🎓 AI Student Assistant with RAG

An intelligent, context-aware student chatbot built for **Sapthagiri NPS University** (Department of Artificial Intelligence & Data Science, Semester III, Section 3ADS01).

Powered by **Retrieval-Augmented Generation (RAG)**, **Groq LLaMA 3.3**, and **Streamlit**.

---

## 🚀 Key Features

1. **Retrieval-Augmented Generation (RAG)**:
   - Uses **TF-IDF Vector Space Model** with unigram + bigram feature extraction and sublinear term frequency.
   - Computes **Cosine Similarity** to retrieve the most mathematically relevant document chunks for each query.
   - Eliminates hallucinations by strictly grounding answers in verified college documents.

2. **Multi-Format Document Support**:
   - Out of the box indexes `timetable.txt` and `college_info.txt`.
   - Dynamic **PDF / TXT / MD Document Uploader** in the sidebar (upload syllabus, circulars, exam notifications on the fly).

3. **Time-Aware Intelligence**:
   - Integrates live Indian Standard Time (`Asia/Kolkata`).
   - Understands relative scheduling queries: *"What is my next class?"*, *"Do we have lab tomorrow?"*, *"Any classes on Saturday?"*.

4. **Transparent Citations**:
   - Shows an expandable **"🔍 View Retrieved Sources (RAG Citations)"** widget under each answer with source names, similarity match scores, and chunk excerpts.

---

## 🛠️ Architecture & Workflow

```
[ User Query ]
       │
       ▼
[ Query Preprocessing & Synonym Expansion ]
       │
       ▼
[ Vector Space Retrieval (Cosine Similarity) ] ◄─── [ Indexed Chunks (PDFs/Text) ]
       │
       ▼
[ Top-K Context Injection + Indian Time Prompt ]
       │
       ▼
[ Groq High-Speed LLM (LLaMA 3.3 70B / 8B) ]
       │
       ▼
[ Response + Interactive RAG Source Citations ]
```

---

## 📦 Setup & Installation

1. **Clone or open the project folder**:
   ```bash
   cd AI_Chatbot_For_Student
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your Groq API Key**:
   - Add your key to `.streamlit/secrets.toml`:
     ```toml
     GROQ_API_KEY = "gsk_your_groq_api_key_here"
     ```
   - *Or* paste it into the secure sidebar input box when running the app.

4. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## 🎓 Viva Questions & Answers

1. **Q: What is RAG and why is it used here?**
   - **A**: RAG (Retrieval-Augmented Generation) retrieves relevant chunks from our local knowledge base (timetable, policies, uploaded PDFs) and passes them into the LLM prompt. This prevents hallucination and keeps the bot updated without retraining.

2. **Q: How does the retrieval algorithm work?**
   - **A**: Documents are segmented into overlapping chunks (500 characters, 80 character overlap). We vectorize the text using TF-IDF (Term Frequency - Inverse Document Frequency) with unigrams and bigrams. The query is vectorized and compared against all document chunks using Cosine Similarity:
     $$\text{Cosine Similarity}(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}$$

3. **Q: Why Groq?**
   - **A**: Groq uses Language Processing Units (LPUs) providing near-instantaneous token generation speed (up to 300+ tokens/sec) and state-of-the-art reasoning with LLaMA 3.3 70B.
