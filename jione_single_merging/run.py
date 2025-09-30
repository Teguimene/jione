from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,  pipeline
from src.workflows import run_task
import torch

def main():
    REV_PHI_128k="072cb7562cb8c4adf682a8e186aaafa49469eb5d"
    REV_PHI_4k="0a67737cc96d2554230f90338b163bc6380a2a85"
    
    print('----------------------------- Starting model downloading ---------------------')

    print('----------------------------- Download Qwen model ---------------------')
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-1.8B",
        torch_dtype="auto",
        # device_map="cuda:0"
    )

    print('----------------------------- Download DistilBert model ---------------------')
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        device=-1  
    )

    print('----------------------------- Download Phi-128k model ---------------------')
    phi_128k_mod = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct", 
        device_map="auto", 
        revision=REV_PHI_128k,
        torch_dtype="auto", 
        trust_remote_code=True
    )
    phi_128k_tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    print('----------------------------- Download Phi-4k model ---------------------')
    phi_4k_mod = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", 
        device_map="auto", 
        revision=REV_PHI_4k,
        torch_dtype="auto", 
        trust_remote_code=True
    )
    phi_4k_tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    
    print('----------------------------- Starting predictions ---------------------')
    # Runin for phi_128k teacher and phi_4k student
    run_task(phi_128k_mod, phi_128k_tok, phi_4k_mod, phi_4k_tok)
    
if __name__ == "__main__":
    main()
