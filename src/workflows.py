import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import  Subset
from typing import Dict, Any, List
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.data_loader import *
from src.evaluation import *
from src.prediction import phi_qwen_prediction, JIONE
from src.prompts import *


def run_task(teacher, student):
    # Datasets paths
    truthfulqa_path = "datasets/truthful_qa_validation_split.jsonl"
    mbpp_path = "datasets/mbpp_test_split.jsonl"
    gsm8k_path = "datasets/gsm8k_test_split.jsonl"
    sst2_path = "datasets/imdb_subset.jsonl"

    t_id = teacher.get("id")
    s_id = student.get("id")
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
        score = 0.0

        pbar = tqdm(task_loader, desc=f"Prediction process for {task} task")
        for i, batch in enumerate(pbar):
            prediction = JIONE(teacher, student, batch, task=task)
            batch['prediction'] = prediction

            # print("prediction", prediction)

            all_batches.extend(convert_dict_of_lists_to_list_of_dicts(batch))

            if torch.cuda.is_available() and (i % 50 == 0):
                torch.cuda.empty_cache()
                
        elapsed = pbar.format_dict["elapsed"]   # en secondes
        hours, rem = divmod(int(elapsed), 3600)
        minutes, _ = divmod(rem, 60)
        elapsed_str = f"{hours:02d}:{minutes:02d}" 
        
        if(t_id != s_id):
            # 4) Save predictions of the task
            dp_df = pd.DataFrame(all_batches)
            dp_df["dummy_field"] = 0

            # créer le dossier s'il n'existe pas
            os.makedirs("outputs", exist_ok=True)

            out_path = f"outputs/{task}_{t_id}(t)_{s_id}(s)_prediction_{elapsed_str}.csv"
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

def iterate_JIONE(
    MODELS: Dict[str, Dict[str, Any]],
    already_done_models: List[str] = None,
) -> None:
    if already_done_models is None:
        already_done_models = []

    model_keys = list(MODELS.keys())

    for teacher_key in model_keys:
        # skip the teacher already done
        if teacher_key in already_done_models:
            print(f"[SKIP] Teacher {teacher_key} already done. \n")
            continue

        teacher = MODELS[teacher_key]

        for student_key in model_keys:
            if student_key == teacher_key:
                continue

            student = MODELS[student_key]

            t_mod = teacher.get("mod")
            s_mod = student.get("mod")

            if t_mod is None or s_mod is None:
                print(f"[WARN] Missing model: teacher_mod={t_mod}, student_mod={s_mod}. \n")
                continue

            print(f"[RUN] {teacher_key}  ==>  {student_key}")
            try:
                run_task(teacher, student)
            except Exception as e:
                print(f"[ERROR] {teacher_key} -> {student_key}: {e} \n")

        already_done_models.append(teacher_key)
        print(f"[DONE] JIONE with {teacher_key} as teacher completed. \n")