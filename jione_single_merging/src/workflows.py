import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import  Subset
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data_loader import *
from src.evaluation import *
from src.prediction import phi_qwen_prediction, JIONE
from src.prompts import *

def run_task(teacher_mod, teacher_tok, student_mod, student_tok):
    # Dataset paths
    truthfulqa_path = "datasets/truthful_qa_validation_split.jsonl"
    mbpp_path = "datasets/mbpp_test_split.jsonl"
    gsm8k_path = "datasets/gsm8k_test_split.jsonl"
    sst2_path = "datasets/imdb_subset.jsonl"

    tasks_order = []
    for t, _ in multi_dataset.samples:
        if t not in tasks_order:
            tasks_order.append(t)

    for task in tasks_order:
        task_indices = [i for i, (t, _) in enumerate(multi_dataset.samples) if t == task]
        task_subset = Subset(multi_dataset, task_indices)
        # data loading
        task_loader = data.DataLoader(
            task_subset,
            batch_size=1,
            num_workers=0,
            shuffle=False,
            collate_fn=collate_fn
        )

        all_batches = [] 

        for i, batch in enumerate(tqdm(task_loader, desc=f"Prediction process for {task} task")):
            prediction = JIONE(teacher_mod, teacher_tok, student_mod, student_tok, batch, task=task)
            batch['prediction'] = prediction

            # print("prediction", prediction)

            all_batches.extend(convert_dict_of_lists_to_list_of_dicts(batch))

            if torch.cuda.is_available() and (i % 50 == 0):
                torch.cuda.empty_cache()

        # 4) Save predictions of the task
        dp_df = pd.DataFrame(all_batches)
        dp_df["dummy_field"] = 0

        # créer le dossier s'il n'existe pas
        os.makedirs("outputs", exist_ok=True)

        out_path = f"outputs/{task}_phi128k(t)_phi4k(s)_prediction.csv"
        dp_df.to_csv(out_path, columns=["id", "prediction", "dummy_field"], index=False)

        if task == "truthful_qa":
            score = Accuracy(out_path, truthfulqa_path)
        elif task == "mbpp":
            score = ROUGE(out_path, mbpp_path)
        elif task == "gsm8k":
            score = EMA(out_path, gsm8k_path)
        else:
            score = evaluate_sst2(out_path, sst2_path)

        print(f"[OK] Saved {len(dp_df)} rows for task '{task}' to {out_path} \n")
        print(f"[EVAL] '{task}' ==> {score} \n")
