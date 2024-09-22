import re
import json
import yaml
import torch
import inspect
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union


def logg(x):
    print(f"------------------------ {x} ---------------------------")

def inspectt(frame):
    logg("")
    args, _, _, values = inspect.getargvalues(frame)
    for arg in args:
        print(f"\t{arg}: {values[arg]}")
    logg("")


def get_prompts_from_template(filepath, name, eval_name):
    default_config = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
    }
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)

    candidate_prompt = data[name]["candidate_prompt"]
    evaluator_prompt = data[eval_name]["evaluator_prompt"]
    candidate_generation_config = data[name].get("candidate_generation_config", default_config)
    evaluator_generation_config = data[eval_name].get(
        "evaluator_generation_config", default_config
    )

    print("candidate_prompt: ", candidate_prompt)
    print("evaluator_prompt: ", evaluator_prompt)
    print("candidate_generation_config: ", candidate_generation_config)
    print("evaluator_generation_config: ", evaluator_generation_config)

    return (
        candidate_prompt,
        evaluator_prompt,
        candidate_generation_config,
        evaluator_generation_config,
    )


def get_tokenizer_and_model(model_name: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/tokenizer",
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/model",
        torch_dtype=torch.float16,
        device_map="auto",
        offload_buffers=True,
    )
    return tokenizer, model


def eval_tokenize(prompt, tokenizer):
    tokenized = tokenizer(prompt, return_tensors="pt")
    return tokenized


def eval_generate_and_tokenize_prompt(data_point, tokenizer, prompt=None):
    prompt = prompt.format(data_point["instruction"], data_point["input"])
    tokenized_full_prompt = eval_tokenize(prompt, tokenizer=tokenizer)
    return tokenized_full_prompt


def eval_prompt_tokenizer(output: str, generated: str, eval_tokenizer, prompt: str = None):
    prompt = prompt.format(output, generated)
    tokenized_full_prompt = eval_tokenize(prompt, tokenizer=eval_tokenizer)
    return tokenized_full_prompt


def extract_score(text):
    match = re.search(r"\b\d+(\.\d+)?\b", text)
    return float(match.group(0)) if match else -1.0


def extract_digit(text):
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def log2json(results, json_result):
    with open(json_result, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def generate_response(
    model, tokenizer, input_ids, attention_mask, generation_config
) -> Union[str, None]:
    try:
        output = model.generate(
            input_ids=torch.LongTensor(input_ids).to(model.device),
            attention_mask=torch.LongTensor(attention_mask).to(model.device),
            eos_token_id=tokenizer.eos_token_id,
            **generation_config,
        )
        response_ids = output[0][len(input_ids[0]) :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        return response
    except RuntimeError as e:
        if "inf" in str(e) or "nan" in str(e):
            print(f"Skipping example due to invalid output: {e}")
            return None
        else:
            raise
