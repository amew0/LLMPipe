# Fine-Tuning Script - README

## Overview

This repository provides a script to fine-tune language models using Hugging Face’s `transformers` and `peft` libraries. To ease up running it, it is set up with semi automatic checkpointing and restarting, configurable hyperparameters, and prompt formatting.

## File Structure

- **`ft.py`**: The main entry point for running the fine-tuning process.
- **`utils/run.py`**: Contains the core logic for loading data, configuring the model, and training.
- **`utils/ft_helper.py`**: Provides utility functions such as logging, resuming from checkpoints, and dataset manipulation.

## Requirements

1. **Environment Variables**: Use `.env` to store environment variables since the `load_dotenv` loads them directly. 
   - `HF_TOKEN_WRITE`: Hugging Face token required to authenticate and upload models or logs.

2. **Dependencies**: Install the required libraries by running:
   ```bash
   [TODO]
   pip install -r requirements.txt
   ```

3. **WandB**: Automatically run logs are reported to WandB and can be turned off by uncommenting the `report_to=None` in `run.py` `line 156`.

## How to Use

### 1. Configuration

Before running the fine-tuning script, ensure you have a `tuning.yaml` file that specifies model-specific configurations such as generation parameters, training hyperparameters, and LoRA configuration.

An example `tuning.yaml`:
```yaml
EleutherAI/pythia-70m-deduped:
  generation_config:
    max_length: 512
    temperature: 0.7
    top_p: 0.9
  training_args:
    learning_rate: 3e-5
    num_train_epochs: 3
  peft_args:
    r: 16
    alpha: 32
    dropout: 0.1
  prompt: "### User:\n{0}\n### Assistant:"
  response: "\n{0}"
```

- **`generation_config`**: Controls the model's generation behavior (e.g., max length, temperature).
- **`training_args`**: Contains the training hyperparameters such as learning rate and epochs.
- **`peft_args`**: Defines the LoRA configuration, including rank (`r`), scaling factor (`alpha`), and dropout.

### 2. Running the Fine-Tuning Script

To fine-tune the model, simply run the `ft.py` script. You can pass additional arguments through the command line:

```bash
python ft.py --model_name EleutherAI/pythia-70m-deduped --train_data_path path_to_data
```

#### Main Arguments:
- **`model_name`**: Pre-trained model to fine-tune.
- **`train_data_path`**: Path to the training dataset, either local or a Hugging Face dataset identifier.
- **`cache_dir`**: (Optional) Directory for caching models, tokenizers, and datasets.
- **`run_id`**: (Optional) Unique identifier for the training run.
- **`model_save_path`**: (Optional) Path to save the fine-tuned model.
- **`chpt_dir`**: (Optional) Directory for saving checkpoints.

#### Example:
```bash
python ft.py --model_name EleutherAI/pythia-70m-deduped --train_data_path my_data --model_save_path output_model
```

### 3. Customization

#### a. **Resuming from Checkpoint**

The fine-tuner can resume from a previous checkpoint. If the `chpt_dir` contains checkpoints, the script will automatically load the latest checkpoint and continue training from the saved state.

#### b. **LoRA Customization**

LoRA parameters (`r`, `alpha`, `dropout`) can be customized in the `tuning.yaml` file under `peft_args`. These parameters control how the low-rank adaptation is applied to the pre-trained model, allowing for more efficient fine-tuning.

#### c. **Prompt Formatting**

The `formatting_prompts_func` in `run.py` allows you to define how input examples are formatted into prompts for the model. Modify the `prompt` and `response` templates in `tuning.yaml` to change how the model interacts with data.

#### d. **Model Generation Config**

Customize how the model generates responses by editing the `generation_config` section in `tuning.yaml`. Options like `temperature` and `max_length` control the diversity and length of the generated text.

### 4. Saving the Model

After training, the fine-tuned model is saved using the Hugging Face format. The output directory can be customized by specifying `model_save_path` when running the script.

```bash
python ft.py --model_save_path /path/to/save
```

The model can be loaded later via:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained('/path/to/save')
```

### 5. Logging and Monitoring

Logs and training metrics can be viewed via print statements or integrated into Weights & Biases (WandB) for better tracking.

## Notes

- **Checkpointing**: The script automatically saves checkpoints at regular intervals to resume training later.
- **Batch Size and Gradient Accumulation**: Modify these values in `run.py` or via `tuning.yaml` to fit your hardware limitations.
- **Distributed Training**: If needed, distributed training parameters (`world_size`, `local_rank`) are available in `run.py`. 
