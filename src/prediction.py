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
def predict_batch(text, classifier):
    # print("predict_batch text", text)
    result = classifier(text)[0]
    return result["label"].lower()

############ JION
def JIONE(teacher, student, batch, task="sst-2"):
    mod = teacher.get("mod")
    tok = teacher.get("tok")
    t_task = teacher.get('task')
    prediction = ""
    if task not in TASKS:
        raise ValueError(f"Task '{task}' not supported. Available: {list(TASKS)}")

    cfg = TASKS[task]
    messages = cfg["jione"](student, t_task , batch)

    if t_task == "generation":
        base_args = {
            "return_full_text": False,
            "do_sample": False
        }
        gen_args = {**base_args, **cfg.get("gen_args", {})}

        pipe = pipeline("text-generation", model=mod, tokenizer=tok)
        prediction = pipe(messages, **gen_args)[0]["generated_text"] 
    else:
        prediction = predict_batch(messages, mod)

    return prediction