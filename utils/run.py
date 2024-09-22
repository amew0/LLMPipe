import os
from datetime import datetime
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from trl import SFTTrainer, SFTConfig
from torch.utils.data import SequentialSampler
from typing import Union
from datasets import load_dataset, Dataset, IterableDataset
from utils.ft_helper import inspectt


class Run:
    def __init__(
        self,
        cache_dir: str,
        train_data_path: str,
        model_name: str,
        model_save_path: str,
        run_id: str,
        chpt_dir: str,
        last_checkpoint: str,
        start_index: int,
        cutoff_len: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        world_size: int,
        local_rank: int,
        verbose: bool,
    ):
        self.cache_dir = cache_dir
        self.train_data_path = train_data_path
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.run_id = run_id
        self.chpt_dir = chpt_dir
        self.last_checkpoint = last_checkpoint
        self.start_index = start_index
        self.cutoff_len = cutoff_len
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.world_size = world_size
        self.local_rank = local_rank
        self.verbose = verbose

        if self.model_save_path is None:
            self.model_save_path = (
                f"{self.cache_dir}/model/{self.model_name}-v{self.run_id}"
            )

        if self.chpt_dir is None:
            self.chpt_dir = f"{self.cache_dir}/chpt/{self.run_id}"

        if os.path.isdir(self.chpt_dir):
            checkpoints = [
                d for d in os.listdir(self.chpt_dir) if d.startswith("checkpoint-")
            ]
            if checkpoints:
                self.last_checkpoint = os.path.join(
                    self.chpt_dir,
                    max(checkpoints, key=lambda cp: int(cp.split("-")[-1])),
                )

        if self.verbose:
            inspectt(locals())

    def load_data(self) -> Union[Dataset, IterableDataset]:
        if self.train_data_path is None:
            raise ValueError("`train_data_path` should be defined!")

        if os.path.exists(self.train_data_path):
            return load_dataset(
                "json",
                data_files=self.train_data_path,
                split="train",
                cache_dir=f"{self.cache_dir}/datasets",
            )

        else:
            return load_dataset(
                self.train_data_path,
                split="train",
                cache_dir=f"{self.cache_dir}/datasets",
            )

    def configure_model(self):
        if not torch.cuda.is_available():
            device_map = "cpu"
            torch_dtype = torch.float32
            bnb_config = None
        else:
            device_map = {"": 0}
            torch_dtype = torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=f"{self.cache_dir}/model",
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        return model

    def load_tokenizer(self, model):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=f"{self.cache_dir}/tokenizer"
        )
        if tokenizer.pad_token is None:
            print("Tokenizer has no pad token. Adding <|pad|> as a pad token.")
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
        return tokenizer

    def get_training_args(self, ft_config, last_checkpoint):
        training_args = ft_config["training_args"]
        train_config = SFTConfig(
            run_name=f"ft-{self.model_name.split('/')[1]}-{self.run_id}-v{self.start_index}",
            resume_from_checkpoint=last_checkpoint,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            output_dir=f"{self.chpt_dir}",
            max_seq_length=self.cutoff_len,  # Truncation/padding under the hood
            eval_accumulation_steps=1,  # Important for data transfer to CPU
            # report_to=None,
            **training_args,
        )
        return train_config

    def trainer_class(self):
        run = self

        class SFTTrainerNoShuffle(SFTTrainer):
            def training_step(self, model, inputs):
                if (self.state.global_step % self.args.save_steps) and run.verbose == 0:
                    inputs_decoded = run.tokenizer.decode(inputs["input_ids"][0])
                    print(f"Step {self.state.global_step}: {inputs_decoded!r}")
                return super().training_step(model, inputs)

            def _get_train_sampler(self):
                return SequentialSampler(self.train_dataset)  # Prevent shuffling

        return SFTTrainerNoShuffle

    def formatting_prompts_func(self, example, ft_config):
        output_texts = []
        for i in range(len(example["instruction"])):
            user_prompt = ft_config["prompt"].format(
                example["instruction"][i], example["input"][i]
            )
            response = ft_config["response"].format(example["output"][i])
            full_prompt = (user_prompt + response).strip()
            output_texts.append(full_prompt)
        return output_texts
