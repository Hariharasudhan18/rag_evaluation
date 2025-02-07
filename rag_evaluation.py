import logging
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import bert_score

def precision_at_k(retrieved_docs, ground_truth, k=3):
    """Computes Precision@k based on relevance of retrieved documents."""
    relevant_count = sum(1 for doc in retrieved_docs[:k] if doc in ground_truth)
    return relevant_count / k

def recall_at_k(retrieved_docs, ground_truth, k=3):
    """Computes Recall@k based on how many relevant documents were retrieved."""
    relevant_count = sum(1 for doc in retrieved_docs[:k] if doc in ground_truth)
    return relevant_count / len(ground_truth) if ground_truth else 0

def mean_reciprocal_rank(retrieved_docs, ground_truth):
    """Computes Mean Reciprocal Rank (MRR) for the retrieved documents."""
    for i, doc in enumerate(retrieved_docs):
        if doc in ground_truth:
            return 1 / (i + 1)
    return 0

def evaluate_retrieval(query, retrieved_docs, ground_truth):
    # Load the pre-trained sentence transformer model for embedding computation
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',cache_folder="C:\\Users\\hari1\\.cache\\huggingface")
    
    # Encode the query and ground truth into vector embeddings
    query_embedding = model.encode(query, convert_to_tensor=True)
    ground_truth_embedding = model.encode(ground_truth, convert_to_tensor=True)
    
    scores = []
    for doc in retrieved_docs:
        # Encode each retrieved document into an embedding
        doc_embedding = model.encode(doc, convert_to_tensor=True)
        
        # Compute cosine similarity between query and retrieved document
        similarity = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
        scores.append(similarity)
    
    # Calculate the average similarity score of the retrieved documents
    avg_similarity = sum(scores) / len(scores) if scores else 0
    
    # Compute similarity between the query and the ground truth
    ground_truth_similarity = util.pytorch_cos_sim(query_embedding, ground_truth_embedding).item()
    
    # Compute additional retrieval metrics
    precision = precision_at_k(retrieved_docs, [ground_truth], k=3)
    recall = recall_at_k(retrieved_docs, [ground_truth], k=3)
    mrr = mean_reciprocal_rank(retrieved_docs, [ground_truth])
    
    # Compute ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(" ".join(retrieved_docs), ground_truth, avg=True)
    
    # Compute BLEU score
    bleu_score = sentence_bleu([ground_truth.split()], retrieved_docs[0].split())
    
    # Compute BERTScore
    P, R, F1 = bert_score.score([retrieved_docs[0]], [ground_truth], model_type='bert-base-uncased')
    bert_f1 = F1.mean().item()
    
    # Log the evaluation results
    logging.info(f"Query: {query}")
    logging.info(f"Retrieved Docs: {retrieved_docs}")
    logging.info(f"Average Similarity with Retrieved Docs: {avg_similarity:.4f}")
    logging.info(f"Similarity with Ground Truth: {ground_truth_similarity:.4f}")
    logging.info(f"Precision@3: {precision:.4f}")
    logging.info(f"Recall@3: {recall:.4f}")
    logging.info(f"MRR: {mrr:.4f}")
    logging.info(f"ROUGE Scores: {rouge_scores}")
    logging.info(f"BLEU Score: {bleu_score:.4f}")
    logging.info(f"BERTScore (F1): {bert_f1:.4f}")
    
    return avg_similarity, ground_truth_similarity, precision, recall, mrr, rouge_scores, bleu_score, bert_f1

if __name__ == "__main__":
    # Configure logging to display evaluation results
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Example query for retrieval evaluation
    query = "What are the benefits of AI in healthcare?"
    
    # Sample retrieved documents from a hypothetical retrieval system
    retrieved_docs = [
        "AI helps doctors diagnose diseases more accurately.",
        "Machine learning improves patient outcomes by predicting risks.",
        "Healthcare chatbots assist in patient inquiries."
    ]
    
    # Ground truth answer for comparison
    ground_truth = "AI enhances healthcare by enabling accurate diagnosis and predicting patient risks."
    
    # Perform retrieval evaluation
    evaluate_retrieval(query, retrieved_docs, ground_truth)
