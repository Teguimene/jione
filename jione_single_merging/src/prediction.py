from src.prompts import *
from tqdm import tqdm
from transformers import  pipeline
######################## Prediction parameters per task ###########
TASKS = {
    'truthful_qa': {
        "ind": truthfulqa_body,
        "jione": jione_truthfulqa_body,
        "gen_args": { "max_new_tokens": 100}
    },
    'mbpp': {
        "ind": mbpp_body,
        "jione": jione_mbpp_body,
        "gen_args": { "max_new_tokens": 300}
    },
    'gsm8k': {
        "ind": gsm8k_body,
        "jione": jione_gsm8k_body,
        "gen_args": { "max_new_tokens": 300}
    },
    'sst-2': {
        "ind": sst2_body,
        "jione": jione_sst2_body,
        "gen_args": { "max_new_tokens": 100}
    },
}

################### Individual prediction function
def phi_qwen_prediction(model, tokenizer, batch, task="sst-2"):
    if task not in TASKS:
        raise ValueError(f"Task '{task}' not supported. Available: {list(TASKS)}")

    cfg = TASKS[task]
    messages = cfg["ind"](batch)

    base_args = {
        "return_full_text": False,
        "do_sample": False
    }
    gen_args = {**base_args, **cfg.get("gen_args", {})}

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    out = pipe(messages, **gen_args)

    return out[0]["generated_text"]


########### distilBERT
def predict_batch(texts, classifier):
    predictions = []
    for text in tqdm(texts, desc="Predicting"):
        result = classifier(text)[0]
        predictions.append(1 if result["label"] == "POSITIVE" else 0)
    return predictions

############ JION
def JIONE(master_mod, master_tok, student_mod, student_tok, batch, task="sst-2"):
    if task not in TASKS:
        raise ValueError(f"Task '{task}' not supported. Available: {list(TASKS)}")

    cfg = TASKS[task]
    messages = cfg["jione"](student_mod, student_tok, batch)

    base_args = {
        "return_full_text": False,
        "do_sample": False
    }
    gen_args = {**base_args, **cfg.get("gen_args", {})}

    pipe = pipeline("text-generation", model=master_mod, tokenizer=master_tok)
    out = pipe(messages, **gen_args)

    return out[0]["generated_text"] 