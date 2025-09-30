import re
import pandas as pd
import json
import os 
import torch
from torch.utils.data import Subset
from torch.utils import data
from typing import List, Dict, Any
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from src.preprocess import extract_final_answer, extract_before_system, extract_sentiment

############################### GSM8K
def EMA(prediction: str, reference: str):
    # csv_path = '/kaggle/working/gsm8k_taskarithmtic(double_prediction)_prediction.csv'
    pred_df = pd.read_csv(prediction)

    # jsonl_path = '/kaggle/input/llm-merging-datasets/gsm8k_test_split.jsonl'
    references = []

    with open(reference, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            references.append(data['output'])

    pred_df = pd.read_csv(prediction)
    predictions = pred_df['prediction'].tolist()

    references = []
    with open(reference, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            references.append(data['output'])

    assert len(predictions) == len(references), "Mismatch: number of predictions ≠ number of references"
    accuracy = 0

    correct_count = 0
    total = len(predictions)
    for pred, ref in tqdm(zip(predictions, references), total=total):
        pred_answer = extract_final_answer(pred)
        ref_answer = extract_final_answer(ref)
        
        if pred_answer is not None and ref_answer is not None:
            if pred_answer == ref_answer:
                correct_count += 1

    accuracy = correct_count / total

    return accuracy

################################### MBPP
def ROUGE(prediction: str, reference: str):
    # csv_path = '/kaggle/input/llm-merging-datasets/mbpp_phi128k(t)_phi4k(s)_prediction_3h40mins.csv'
    pred_df = pd.read_csv(prediction)

    # jsonl_path = '/kaggle/input/llm-merging-datasets/mbpp_test_split.jsonl'
    references = []

    with open(reference, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            references.append(data['output'])

    pred_df = pd.read_csv(prediction)
    predictions = pred_df['prediction'].tolist()

    references = []
    with open(reference, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            references.append(data['output'])

    assert len(predictions) == len(references), "Mismatch: number of predictions ≠ number of references"

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
        pred_answer = extract_before_system(pred)
        ref_answer = extract_before_system(ref)
        scores = scorer.score(pred_answer, ref_answer)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    overall_avg = (avg_rouge1 + avg_rouge2 + avg_rougeL) / 3

    return overall_avg

############################### TruthfulQA
def Accuracy(prediction: str, reference: str):
    # csv_path = '/kaggle/input/llm-merging-datasets/truthful_qa_phi128k(m)_phi4k(s)_prediction_45m20s.csv'
    pred_df = pd.read_csv(prediction)

    # jsonl_path = '/kaggle/input/llm-merging-datasets/truthful_qa_validation_split.jsonl'
    references = []

    with open(reference, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            references.append(data['output'].strip())

    predictions = pred_df['prediction'].apply(lambda x: x.strip() if isinstance(x, str) else "").tolist()

    assert len(predictions) == len(references), "Mismatch: nombre de prédictions ≠ nombre de références"

    #Fonction de normalisation
    def normalize(text):
        return text.strip().rstrip('.').lower()

    #Calcul de l'accuracy
    correct = 0
    total = len(predictions)

    for pred, ref in tqdm(zip(predictions, references), total=total):
        if normalize(pred) == normalize(ref):
            correct += 1

    accuracy = correct / total

    return accuracy

############################### SST-2
def clean_label(text):
    if not isinstance(text, str):
        return text
    return text.strip().strip('"').strip("'").strip("[]").lower()

def evaluate_sst2(pred_file_csv, ref_file_jsonl):
    pred_df = pd.read_csv(pred_file_csv)
    if 'prediction' not in pred_df.columns:
        raise ValueError("The CSV file should contain the 'prediction' column.")

    with open(ref_file_jsonl, 'r', encoding='utf-8') as f:
        labels = [json.loads(line)['label'] for line in f]

    if len(labels) != len(pred_df):
        raise ValueError(f"Number of labels ({len(labels)}) ≠ number of predictions ({len(pred_df)})")

    # Clean datas
    y_true = [clean_label(l.lower()) for l in labels]
    y_pred = [clean_label(extract_sentiment(p.lower())) for p in pred_df['prediction']]

    # Verification of authorized value
    valid_values = {"positive", "negative","neutral"}
    if not all(l in valid_values for l in y_true):
        raise ValueError("The csv file contains invalid labels.")
    if not all(p in valid_values for p in y_pred):
        raise ValueError("The csv file contains invalid predictions.")

    # Evaluate
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy