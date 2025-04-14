# Customer Support AI Assistant

A Customer Support AI Assistant using Retrieval-Augmented Generation (RAG) system with built-in explainability features, evaluation framework,feedback mechanism and user-friendly interface.

![Assistant](images/exampleUI.png)

## Key Features

### RAG Pipeline

- Document retrieval based on semantic similarity
- Context integration with advanced LLM
- Response generation with source attribution in explainability

### Explainability

- Confidence scoring for responses
- Key information extraction from context documents
- Reasoning trace generation
- Faithfulness scoring
- Context highlighting in responses

### Evaluation Framework

- Content-based metrics (ROUGE, BLEU, semantic similarity)
- Retrieval-based metrics (context relevance, utilization)
- Response quality metrics (readability, query coverage)
- Performance metrics (response time)
- Batch evaluation and result saving

## Installation

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/rag-explainability-system.git
cd rag-explainability-system
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
# Edit .env with your configuration
GROQ_API_KEY=
HUGGINGFACEHUB_API_TOKEN=
```

5. Modify the QA data,process and push it in Vector DB

```bash
# Modify the no of documents we need to take for local(according to your local resources)
python -m models.embedding
```

6. Start the Backend

```bash
python -m app.py
```

7. Start the Frotend

```bash
streamlit run ui/streamlit_app.py
```

## Deployment

### Frontend

The frontend is deployed on Hugging Face Spaces:

URL: https://huggingface.co/spaces/yourusername/rag-explainability

Built with Streamlit

### Backend

The backend is deployed on Render:

API Endpoint: https://rag-explainability-backend.onrender.com

Pushed a sample QA embedding Vector space with it.

## Evaluation

```bash
python -m evaluation.evaluation
```
