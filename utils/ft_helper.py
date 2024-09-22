import os
import json
import inspect
from datasets import concatenate_datasets


def logg(x):
    print(f"------------------------ {x} ---------------------------")


def inspectt(kwargs):
    """
    Logs the values of the passed arguments to a function.
    """
    logg("Inspecting")
    for kwarg in kwargs:
        print(f"\t{kwarg}: {kwargs[kwarg]}")

def get_start_index(last_checkpoint, total_rows) -> int:
    """
    Get the start index to resume training from the last checkpoint.

    Args:
        last_checkpoint: str
        total_rows: int
    Returns:
        start_index: int
    """
    with open(os.path.join(last_checkpoint, "trainer_state.json"), "r") as f:
        trainer_state = json.load(f)

    start_index = (total_rows * trainer_state["epoch"]) % total_rows
    return start_index


def reorder_dataset(dataset, start_index):
    # Split the dataset into two parts: before and after the start index
    dataset_part1 = dataset.select(range(start_index, len(dataset)))
    dataset_part2 = dataset.select(range(start_index))

    # Concatenate the two parts to get the reordered dataset
    reordered_dataset = concatenate_datasets([dataset_part1, dataset_part2])
    return reordered_dataset
