# RAG Evaluation

This repository contains a basic implementation of Retrieval-Augmented Generation (RAG) evaluation, specifically focusing on retrieval and generation evaluation using multiple metrics.

## Types of RAG Evaluation Metrics

### **1. Retrieval Metrics**
These evaluate how well the retrieved documents match the query.

#### **Precision@k**
- Measures the fraction of the top-k retrieved documents that are relevant.
- Formula:  
  \[
  \text{Precision@k} = \frac{\text{Relevant Retrieved Documents}}{k}
  \]
- **Strength**: Useful when only the top-k documents matter.
- **Weakness**: Doesn't consider the full list of retrieved documents.

#### **Recall@k**
- Measures how many of the relevant documents were retrieved within the top-k results.
- Formula:  
  \[
  \text{Recall@k} = \frac{\text{Relevant Retrieved Documents}}{\text{Total Relevant Documents}}
  \]
- **Strength**: Important when finding all relevant results is crucial.
- **Weakness**: May favor retrieving more documents, even if some are less relevant.

#### **Mean Reciprocal Rank (MRR)**
- Evaluates how soon the first relevant document appears in the ranked list.
- Formula:  
  \[
  \text{MRR} = \frac{1}{\text{Rank of First Relevant Document}}
  \]
- **Strength**: Rewards systems that retrieve relevant documents early.
- **Weakness**: Doesn't consider additional relevant documents lower in the ranking.

#### **Cosine Similarity**
- Measures the angle between the query and retrieved document vectors.
- Formula:  
  \[
  \text{cos}(\theta) = \frac{A \cdot B}{||A|| ||B||}
  \]
- **Strength**: Works well for text embeddings and semantic similarity.
- **Weakness**: Doesn't consider word order or fluency.

### **2. Generation Metrics**
These assess the **quality of the generated text** compared to the ground truth.

#### **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- Measures overlap between the generated response and the ground truth.
- **Types**:
  - **ROUGE-1**: Unigram (single-word) overlap.
  - **ROUGE-2**: Bigram (two-word) overlap.
  - **ROUGE-L**: Longest common subsequence (LCS).
- **Strength**: Good for summarization and recall-oriented tasks.
- **Weakness**: Doesn't account for synonyms or meaning differences.

#### **BLEU (Bilingual Evaluation Understudy)**
- Measures n-gram overlap between generated text and reference text.
- Formula:  
  \[
  \text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
  \]
  where **BP** (brevity penalty) discourages overly short responses.
- **Strength**: Effective for translation tasks.
- **Weakness**: Doesn't handle synonyms or reworded sentences well.

#### **BERTScore**
- Uses **BERT embeddings** to compute semantic similarity.
- Instead of exact word overlap, it calculates cosine similarity between word embeddings.
- **Strength**: Captures meaning rather than exact words.
- **Weakness**: Requires deep learning models, making it computationally expensive.

## **Which Metric is Better?**
| **Metric** | **Best for** | **Strengths** | **Weaknesses** |
|------------|-------------|---------------|----------------|
| **Precision@k** | Ranked retrieval | Good for top-k relevance | Ignores overall recall |
| **Recall@k** | Comprehensive search | Ensures all relevant results are included | Can favor recall over precision |
| **MRR** | First relevant document ranking | Rewards high-ranking relevant results | Ignores lower-ranked relevant documents |
| **Cosine Similarity** | Semantic similarity | Captures meaning from embeddings | Doesn't consider word order |
| **ROUGE** | Summarization | Measures text overlap well | Doesn't consider meaning |
| **BLEU** | Machine translation | Evaluates fluency | Penalizes rewording |
| **BERTScore** | Natural language understanding | Captures meaning, even with synonyms | High computational cost |

### **Best Choice for RAG Evaluation?**
- **For Retrieval**: **MRR + Cosine Similarity** are good for ranking effectiveness.
- **For Generation**: **BERTScore** is best as it captures meaning rather than just word overlap.

## Running the Evaluation

1. Install dependencies:
   ```bash
   pip install sentence-transformers rouge-score nltk bert-score
   ```
2. Run the script:
   ```bash
   python rag_evaluation.py
   ```

## Repository Structure

- `rag_evaluation.py` - Main script for evaluating retrieval and generation quality.
- `README.md` - Documentation of the approach and evaluation methodology.

