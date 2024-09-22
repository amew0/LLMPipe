import socket
import os
import gc
import fire
import yaml
from time import time
from datetime import datetime
from dotenv import load_dotenv
import huggingface_hub
import wandb
from peft import LoraConfig
from utils.run import Run
from utils.ft_helper import inspectt, logg, get_start_index, reorder_dataset

from datasets.utils.logging import disable_progress_bar

disable_progress_bar()
wandb.require("core")
print(socket.gethostname())


def main(
    cache_dir: str = "/dpc/kunf0097/l3-8b",
    train_data_path: str = "meher146/medical_llama3_instruct_dataset",
    model_name: str = "EleutherAI/pythia-70m-deduped",
    model_save_path: str = None,
    run_id: str = datetime.now().strftime("%y%m%d%H%M%S"),
    chpt_dir: str = None,
    last_checkpoint: str = None,
    start_index: int = 0,
    cutoff_len: int = 512,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    world_size: int = None,
    local_rank: int = None,
    verbose: bool = True,
):
    """
    Finetuning.

    Args:
        cache_dir (str): Directory for caching models/tokenizers/datasets.
        train_data_path (str): Path to training data.
        model_name (str): Name of the model to fine-tune.
        model_save_path (str): Path to save the fine-tuned model.
        run_id (str): Unique identifier for the run.
        chpt_dir (str): Directory for checkpoints.
        last_checkpoint (str): Path to the last checkpoint.
        start_index (int): Start index for the dataset.
        cutoff_len (int): Cutoff length for the dataset (For batching).
        per_device_train_batch_size (int): Batch size per device.
        gradient_accumulation_steps (int): Steps for gradient accumulation.
        world_size (int): Number of distributed processes.
        local_rank (int): Local rank for distributed training.
        verbose (bool): Verbosity.
    """
    run = Run(**locals())

    load_dotenv()
    HF_TOKEN_WRITE = os.getenv("HF_TOKEN_WRITE")
    if HF_TOKEN_WRITE is not None:
        huggingface_hub.login(token=HF_TOKEN_WRITE)

    data = run.load_data()

    if run.last_checkpoint is not None:
        run.start_index = get_start_index(run.last_checkpoint, len(data))

    # Load finetuning configuration
    with open(f"tuning.yaml", "r") as f:
        ft_config = yaml.safe_load(f)[run.model_name]
        print(ft_config)

    # Initialize model
    model = run.configure_model()

    # Apply generation configuration
    generation_config = ft_config["generation_config"]
    for key, value in generation_config.items():
        setattr(model.generation_config, key, value)
    print(model.generation_config)

    # Load tokenizer
    tokenizer = run.load_tokenizer(model)

    # Prepare LoRA
    peft_args = ft_config["peft_args"]
    peft_config = LoraConfig(**peft_args)

    if run.start_index != 0:
        data = reorder_dataset(data, run.start_index)

    # Prepare training configuration
    train_config = run.get_training_args(ft_config, run.last_checkpoint)

    # Trainer class
    SFTTrainerNoShuffle = run.trainer_class()

    from trl import DataCollatorForCompletionOnlyLM

    response_template = "### Assistant:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainerNoShuffle(
        model=model,
        tokenizer=tokenizer,
        formatting_func=lambda example: run.formatting_prompts_func(example, ft_config),
        data_collator=collator,
        peft_config=peft_config,
        train_dataset=data,
        args=train_config,
    )

    # Train model
    gc.collect()
    gc.collect()

    start = time()
    if run.last_checkpoint is not None:
        logg("Resuming from checkpoint")
        trainer.train(run.last_checkpoint)
    else:
        trainer.train()
    end = time()
    logg(f"Elapsed time: {end - start}")

    trainer.model.save_pretrained(run.model_save_path)


if __name__ == "__main__":
    logg("ft-medical.py")
    fire.Fire(main)
