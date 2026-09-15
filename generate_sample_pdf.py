"""
generate_sample_pdf.py
Creates sample_data/sample_document.pdf — a 6-page educational document on
Artificial Intelligence fundamentals.  Requires PyMuPDF (fitz).
Run: python generate_sample_pdf.py
"""
import sys
from pathlib import Path

try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed. Run: pip install pymupdf")
    sys.exit(1)

OUTPUT_DIR = Path("sample_data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "sample_document.pdf"

# ---------------------------------------------------------------------------
# Page content (6 pages)
# ---------------------------------------------------------------------------
PAGES = [
    {
        "title": "Introduction to Artificial Intelligence",
        "body": (
            "Artificial Intelligence (AI) is a branch of computer science dedicated to building systems "
            "that can perform tasks that typically require human intelligence. These tasks include "
            "reasoning, learning, problem-solving, perception, and language understanding.\n\n"
            "The term 'Artificial Intelligence' was coined by John McCarthy in 1956 at the Dartmouth "
            "Conference, which is widely regarded as the founding event of AI as a formal discipline. "
            "Since then, AI has evolved through multiple periods of excitement and stagnation — commonly "
            "referred to as 'AI summers' and 'AI winters'.\n\n"
            "Modern AI is primarily driven by machine learning techniques, particularly deep learning, "
            "which have produced remarkable breakthroughs in image recognition, natural language "
            "processing, game-playing, and scientific discovery."
        ),
    },
    {
        "title": "Machine Learning Fundamentals",
        "body": (
            "Machine Learning (ML) is a subfield of AI that enables systems to learn from data and "
            "improve their performance without being explicitly programmed. Rather than writing rules "
            "by hand, ML algorithms discover patterns in data and generalise them to new examples.\n\n"
            "There are three primary learning paradigms in machine learning:\n\n"
            "1. Supervised Learning — The model is trained on labelled data (input-output pairs). "
            "Common tasks include classification (e.g., spam detection) and regression (e.g., house "
            "price prediction).\n\n"
            "2. Unsupervised Learning — The model discovers hidden structure in unlabelled data. "
            "Clustering algorithms (e.g., k-means) and dimensionality reduction techniques "
            "(e.g., PCA) fall into this category.\n\n"
            "3. Reinforcement Learning — An agent learns by interacting with an environment and "
            "receiving reward signals. This paradigm has produced world-class performance in games "
            "like chess and Go."
        ),
    },
    {
        "title": "Deep Learning and Neural Networks",
        "body": (
            "Deep learning is a subset of machine learning that uses artificial neural networks with "
            "many layers — hence the term 'deep'. Inspired loosely by the biological structure of the "
            "human brain, a neural network consists of layers of interconnected nodes (neurons), each "
            "applying a non-linear transformation to its inputs.\n\n"
            "Key components of a neural network:\n\n"
            "- Input Layer: Receives raw features (e.g., pixel values of an image).\n"
            "- Hidden Layers: Apply learned transformations to extract progressively abstract features.\n"
            "- Output Layer: Produces the final prediction (e.g., class probabilities).\n"
            "- Activation Functions: Introduce non-linearity (e.g., ReLU, sigmoid, softmax).\n"
            "- Backpropagation: The algorithm that adjusts weights using gradient descent.\n\n"
            "Convolutional Neural Networks (CNNs) excel at image tasks; Recurrent Neural Networks "
            "(RNNs) and Transformers dominate sequence modelling and natural language processing. "
            "The Transformer architecture, introduced in 2017 by Vaswani et al., is now the "
            "foundation of large language models such as GPT and Gemini."
        ),
    },
    {
        "title": "Natural Language Processing",
        "body": (
            "Natural Language Processing (NLP) is the subfield of AI concerned with enabling computers "
            "to understand, interpret, and generate human language. NLP bridges linguistics and machine "
            "learning to handle the ambiguity, context-dependence, and richness of natural language.\n\n"
            "Core NLP tasks include:\n\n"
            "- Tokenisation: Splitting text into words or subword units.\n"
            "- Part-of-Speech Tagging: Identifying grammatical roles of words.\n"
            "- Named Entity Recognition (NER): Detecting names, dates, locations in text.\n"
            "- Sentiment Analysis: Classifying the emotional tone of text.\n"
            "- Machine Translation: Automatically translating between languages.\n"
            "- Question Answering: Locating or generating answers from documents.\n"
            "- Text Summarisation: Producing concise summaries of long documents.\n\n"
            "Large Language Models (LLMs) such as GPT-4 and Gemini are trained on massive text corpora "
            "and can perform all of the above tasks with high accuracy using a single, general-purpose "
            "architecture — the Transformer."
        ),
    },
    {
        "title": "AI Applications in the Real World",
        "body": (
            "Artificial intelligence is now deeply embedded in everyday life and high-stakes industries:\n\n"
            "Healthcare: AI assists in medical image analysis (detecting tumours in X-rays and MRIs), "
            "drug discovery, genomic sequencing interpretation, and clinical decision support. AlphaFold "
            "by DeepMind predicted the structure of nearly every known protein — a scientific milestone.\n\n"
            "Finance: Fraud detection systems process millions of transactions in real-time to flag "
            "anomalies. Algorithmic trading, credit scoring, and risk assessment are all driven by ML.\n\n"
            "Transportation: Autonomous vehicles from companies such as Waymo and Tesla rely on "
            "computer vision, sensor fusion, and reinforcement learning to navigate complex environments.\n\n"
            "Education: Personalised learning platforms adapt content to individual student needs. "
            "AI tutors can answer student questions, grade essays, and identify learning gaps.\n\n"
            "Customer Service: Conversational AI chatbots handle millions of support queries, reducing "
            "wait times and costs while maintaining consistent responses 24 hours a day."
        ),
    },
    {
        "title": "Ethics, Safety, and the Future of AI",
        "body": (
            "As AI systems become more capable and pervasive, ethical and safety considerations have "
            "moved to the forefront of the field.\n\n"
            "Bias and Fairness: ML models trained on historical data can perpetuate or amplify "
            "societal biases. Facial recognition systems, hiring algorithms, and credit models have "
            "all exhibited documented bias against minority groups.\n\n"
            "Transparency and Explainability: Many deep learning models are 'black boxes'. Efforts in "
            "Explainable AI (XAI) aim to make model decisions interpretable to humans — critical in "
            "healthcare and legal contexts.\n\n"
            "Privacy: Large-scale data collection needed to train AI systems raises serious privacy "
            "concerns. Differential privacy and federated learning are emerging techniques to train "
            "models without centralising sensitive data.\n\n"
            "AI Safety: The long-term goal of ensuring that highly capable AI systems remain aligned "
            "with human values and intentions is an active area of research, with organisations such "
            "as DeepMind, OpenAI, and Anthropic publishing safety frameworks.\n\n"
            "The Future: Researchers are pursuing Artificial General Intelligence (AGI) — systems "
            "that can perform any intellectual task a human can. While timelines are debated, current "
            "progress in reasoning, planning, and multi-modal AI suggests the field will continue its "
            "rapid advancement throughout the coming decades."
        ),
    },
]

# ---------------------------------------------------------------------------
# Build PDF with PyMuPDF
# ---------------------------------------------------------------------------
def create_pdf(output_path: Path):
    doc = fitz.open()  # New empty document

    for page_num, page_data in enumerate(PAGES, start=1):
        page = doc.new_page(width=612, height=792)  # US Letter

        # ---- Header bar ----
        page.draw_rect(fitz.Rect(0, 0, 612, 50), color=(0.15, 0.25, 0.5), fill=(0.15, 0.25, 0.5))
        page.insert_text(
            (20, 32),
            "GA_7_SecondBrain_BYTE — Artificial Intelligence Primer",
            fontsize=10,
            color=(1, 1, 1),
        )

        # ---- Chapter title ----
        page.insert_text(
            (40, 90),
            f"Chapter {page_num}: {page_data['title']}",
            fontsize=16,
            color=(0.1, 0.2, 0.45),
        )

        # ---- Horizontal rule ----
        page.draw_line((40, 105), (572, 105), color=(0.7, 0.7, 0.7), width=1)

        # ---- Body text (word-wrap via textbox) ----
        text_rect = fitz.Rect(40, 120, 572, 730)
        page.insert_textbox(
            text_rect,
            page_data["body"],
            fontsize=11,
            color=(0.1, 0.1, 0.1),
            align=fitz.TEXT_ALIGN_LEFT,
        )

        # ---- Footer ----
        page.draw_line((40, 755), (572, 755), color=(0.7, 0.7, 0.7), width=0.5)
        page.insert_text(
            (40, 770),
            f"Page {page_num} of {len(PAGES)}  |  Sample Educational Document  |  BYTE Arithmatrix AVIP 2026",
            fontsize=8,
            color=(0.5, 0.5, 0.5),
        )

    doc.save(str(output_path))
    doc.close()
    print(f"Created: {output_path}  ({len(PAGES)} pages)")


if __name__ == "__main__":
    create_pdf(OUTPUT_PATH)
