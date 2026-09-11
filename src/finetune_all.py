import math
import subprocess
import json
import sys
import typer

from collections import defaultdict
from typing import Annotated, List, Literal, Tuple


def main(
        checkpoint: Annotated[str, typer.Option()],
        model_type: Annotated[Literal["latentwfm", "wavesfm"], typer.Option()],
        checkpoint_key_prefix: Annotated[str, typer.Option()],
        tasks: Annotated[list[str], typer.Option()] = ["rfs", "pos", "deepmimo-beam", "deepmimo-los"],
        methods: Annotated[list[str], typer.Option()] = ["lp", "lora", "ft2"],
        seeds: Annotated[list[int], typer.Option()] = [0, 1, 2]
    ):
    
    method_args = {
        "lp": [],
        "lora": ["--lora", "--lora-rank", "32", "--lora-alpha", "64"],
        "ft2": ["--frozen-blocks", "6"]
    }

    tasks_config = {
        "rfs": {
            "dataset_path": "../preprocessed_datasets/rfs.h5",
            "additional_args": [
                "--stratified-split",
            ],
            "batch_size": "128"
        },
        "pos": {
            "dataset_path": "../preprocessed_datasets/nrpos-outdoor.h5",
            "additional_args": [],
            "batch_size": "128"
        },
        "deepmimo-beam": {
            "dataset_path": "../preprocessed_datasets/deepmimo.h5",
            "additional_args": [
                "--stratified-split",
                "--class-weights",
                "--vis-img-size", "32",
                "--deepmimo-n-beams", "16"
            ],
            "batch_size": "256"
        },
        "deepmimo-los": {
            "dataset_path": "../preprocessed_datasets/deepmimo.h5",
            "additional_args": [
                "--class-weights",
                "--vis-img-size", "32"
            ],
            "batch_size": "256"
        }
    }

    method_task_args = defaultdict(lambda: defaultdict(list))
    method_task_args["lp"]["rfs"] = ["--weight-decay", "0.1", "--lr", "1e-2"]

    checkpoint_name = "_".join(checkpoint.split("/")[-1].split("-")[:4])
    output_path = f"runs/{checkpoint_name}/finetune_results.json"
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for method in methods:
        for task in tasks:
            config = tasks_config[task]
            for seed in seeds:
                print(f"Running {method}: {task} with seed {seed}...", flush=True)
                try:
                    result = subprocess.run([
                        "uv", "run", "src/finetune.py",
                        "--task", task,
                        "--train-data", config["dataset_path"],
                        "--val-split", "0.2",
                        "--output-dir", f"runs/{checkpoint_name}/{task}/{method}/seed_{seed}",
                        "--finetune", checkpoint,
                        "--model", "vit_multi_small",
                        "--model-type", model_type,
                        "--checkpoint-key-prefix", checkpoint_key_prefix,
                        "--freeze-tokenizer",
                        "--warmup-epochs", "5",
                        "--num-workers", "2",
                        "--seed", str(seed),
                        "--epochs", "100",
                        "--batch-size", config["batch_size"],
                        "--layer-decay", "1",
                        *config["additional_args"],
                        *method_args[method],
                        *method_task_args[method][task]
                    ], capture_output=True, text=True)

                    if result.returncode != 0:
                        print(f"ERROR: {task} seed {seed} failed with return code {result.returncode}")
                        print(result.stdout if result.stdout else "No stdout")
                        print(result.stderr if result.stderr else "No stderr")
                        sys.exit(1)

                    last_line = result.stdout.strip().splitlines()[-1]
                    perf = float(last_line.split("=")[-1])
                    results[method][task]["perfs"].append(perf)

                    with open(output_path, "w") as f:
                        json.dump(results, f, indent=4)

                except Exception as e:
                    print(f"ERROR: Exception during {task} seed {seed}: {e}")
                    sys.exit(1)

            mean, std = mean_std(results[method][task]["perfs"])
            results[method][task]["mean"] = mean
            results[method][task]["std"] = std
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)


def mean_std(values: List[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n  
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var)


if __name__ == "__main__":
    typer.run(main)
