# Sample Q&A — GA_7_SecondBrain_BYTE

> **Document indexed:** `sample_data/sample_document.pdf` (6 pages)
> **Embedding model:** `all-MiniLM-L6-v2` (local, offline)
> **LLM:** `gemini-2.5-flash`
>
> All answers below are strictly grounded in the retrieved document context.  
> Page citations reference the exact pages where supporting text appears.

---

## Q1: What is Artificial Intelligence and who coined the term?

**Answer:**

Artificial Intelligence (AI) is a branch of computer science dedicated to building systems that can perform tasks that typically require human intelligence — including reasoning, learning, problem-solving, perception, and language understanding.

The term "Artificial Intelligence" was coined by **John McCarthy in 1956** at the **Dartmouth Conference**, which is widely regarded as the founding event of AI as a formal discipline. [Page 1]

**Verified Citations:** Page 1  
**Retrieved from:** Chapter 1 — Introduction to Artificial Intelligence

---

## Q2: What are the three primary learning paradigms in machine learning?

**Answer:**

According to the document, the three primary learning paradigms in machine learning are: [Page 2]

1. **Supervised Learning** — The model is trained on labelled data (input-output pairs). Common tasks include classification (e.g., spam detection) and regression (e.g., house price prediction). [Page 2]

2. **Unsupervised Learning** — The model discovers hidden structure in unlabelled data. Clustering algorithms (e.g., k-means) and dimensionality reduction techniques (e.g., PCA) fall into this category. [Page 2]

3. **Reinforcement Learning** — An agent learns by interacting with an environment and receiving reward signals. This paradigm has produced world-class performance in games like chess and Go. [Page 2]

**Verified Citations:** Page 2  
**Retrieved from:** Chapter 2 — Machine Learning Fundamentals

---

## Q3: What is a neural network and what are its key components?

**Answer:**

A neural network consists of layers of interconnected nodes (neurons), each applying a non-linear transformation to its inputs. It is inspired loosely by the biological structure of the human brain. [Page 3]

The key components include: [Page 3]

- **Input Layer** — Receives raw features (e.g., pixel values of an image)
- **Hidden Layers** — Apply learned transformations to extract progressively abstract features
- **Output Layer** — Produces the final prediction (e.g., class probabilities)
- **Activation Functions** — Introduce non-linearity (e.g., ReLU, sigmoid, softmax)
- **Backpropagation** — The algorithm that adjusts weights using gradient descent

**Verified Citations:** Page 3  
**Retrieved from:** Chapter 3 — Deep Learning and Neural Networks

---

## Q4: What Transformer architecture paper is mentioned, and when was it published?

**Answer:**

The document states that the **Transformer architecture was introduced in 2017 by Vaswani et al.** and is now the foundation of large language models such as GPT and Gemini. [Page 3]

**Verified Citations:** Page 3  
**Retrieved from:** Chapter 3 — Deep Learning and Neural Networks

---

## Q5: What are some real-world applications of AI in healthcare?

**Answer:**

According to the document, AI assists in healthcare through: [Page 5]

- **Medical image analysis** — detecting tumours in X-rays and MRIs
- **Drug discovery**
- **Genomic sequencing interpretation**
- **Clinical decision support**

Additionally, **AlphaFold** by DeepMind is highlighted as a milestone: it predicted the structure of nearly every known protein. [Page 5]

**Verified Citations:** Page 5  
**Retrieved from:** Chapter 5 — AI Applications in the Real World

---

## Q6: What is Natural Language Processing and what tasks does it cover?

**Answer:**

Natural Language Processing (NLP) is the subfield of AI concerned with enabling computers to understand, interpret, and generate human language. It bridges linguistics and machine learning to handle the ambiguity, context-dependence, and richness of natural language. [Page 4]

Core NLP tasks listed in the document include: [Page 4]

- **Tokenisation** — Splitting text into words or subword units
- **Part-of-Speech Tagging** — Identifying grammatical roles of words
- **Named Entity Recognition (NER)** — Detecting names, dates, locations in text
- **Sentiment Analysis** — Classifying the emotional tone of text
- **Machine Translation** — Automatically translating between languages
- **Question Answering** — Locating or generating answers from documents
- **Text Summarisation** — Producing concise summaries of long documents

**Verified Citations:** Page 4  
**Retrieved from:** Chapter 4 — Natural Language Processing

---

## Q7: What ethical concerns related to AI does the document discuss?

**Answer:**

The document discusses several key ethical and safety concerns about AI: [Page 6]

- **Bias and Fairness** — ML models trained on historical data can perpetuate or amplify societal biases. Facial recognition systems, hiring algorithms, and credit models have exhibited documented bias against minority groups. [Page 6]
- **Transparency and Explainability** — Many deep learning models are "black boxes". Explainable AI (XAI) aims to make model decisions interpretable, especially in healthcare and legal contexts. [Page 6]
- **Privacy** — Large-scale data collection raises serious privacy concerns. Differential privacy and federated learning are emerging mitigation techniques. [Page 6]
- **AI Safety** — Ensuring that highly capable AI systems remain aligned with human values. Organisations such as DeepMind, OpenAI, and Anthropic publish safety frameworks. [Page 6]

**Verified Citations:** Page 6  
**Retrieved from:** Chapter 6 — Ethics, Safety, and the Future of AI

---

## Q8: What is the difference between machine learning and deep learning?

**Answer:**

Based on the document: [Page 2, Page 3]

- **Machine Learning** is a subfield of AI that enables systems to learn from data and improve their performance without being explicitly programmed. It encompasses multiple paradigms including supervised, unsupervised, and reinforcement learning. [Page 2]

- **Deep Learning** is a *subset* of machine learning that specifically uses artificial neural networks with many layers. The term "deep" refers to the multiple hidden layers in these networks. Deep learning excels at tasks like image recognition and natural language understanding due to its ability to extract hierarchical feature representations. [Page 3]

In short: all deep learning is machine learning, but not all machine learning is deep learning.

**Verified Citations:** Page 2, Page 3  
**Retrieved from:** Chapter 2 — Machine Learning Fundamentals; Chapter 3 — Deep Learning and Neural Networks

---

*These answers were generated by the GA_7_SecondBrain_BYTE RAG system. Each answer uses only information retrieved from the indexed PDF document. No external knowledge was used.*
