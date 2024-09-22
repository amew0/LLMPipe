""""
This file is to connect the trained/ fine tuned adapter with the base model
and upload the adapter and/or the merged model to the HF library.
"""

import os
import fire
from utils.ft_helper import get_start_index
from dotenv import load_dotenv

import huggingface_hub
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(
    cache_dir: str = "/dpc/kunf0097/l3-8b",
    model_name: str = "EleutherAI/pythia-70m-deduped",
    run_id: str = "240730153003",
    chpt_dir: str = None,
    adapter_name: str = None,
    push_adapter_only: bool = True,
    push_unloaded_model: bool = False,
):
    load_dotenv()
    HF_TOKEN_WRITE = os.environ.get("HF_TOKEN_WRITE")
    if HF_TOKEN_WRITE:
        huggingface_hub.login(token=HF_TOKEN_WRITE)

    if chpt_dir is None:
        chpt_dir = f"{cache_dir}/chpt/{run_id}"

    last_checkpoint = None
    if os.path.isdir(chpt_dir):
        checkpoints = [d for d in os.listdir(chpt_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = os.path.join(
                chpt_dir, max(checkpoints, key=lambda cp: int(cp.split("-")[-1]))
            )

    if adapter_name is None:
        adapter_name = f"{model_name.split('/')[1]}-v{run_id}-ada"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=f"{cache_dir}/model",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        last_checkpoint, cache_dir=f"{cache_dir}/tokenizer"
    )

    model.resize_token_embeddings(len(tokenizer))
    ftmodel = PeftModel.from_pretrained(model, last_checkpoint)

    if push_adapter_only:
        ftmodel.push_to_hub(adapter_name, token=HF_TOKEN_WRITE)
        tokenizer.push_to_hub(adapter_name, token=HF_TOKEN_WRITE)

    if push_unloaded_model:
        ftmodel = ftmodel.merge_and_unload()
        tokenizer.push_to_hub(
            adapter_name.split("-")[0],
            token=HF_TOKEN_WRITE,
        )
        ftmodel.push_to_hub(
            adapter_name.split("-")[0],
            token=HF_TOKEN_WRITE,
        )

if __name__ == "__main__":
    fire.Fire(main)
