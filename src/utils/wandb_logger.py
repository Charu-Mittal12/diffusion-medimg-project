"""Thin wrapper around Weights & Biases so every script logs consistently:
full config, metrics, and artifacts (datasets, model weights, result tables).
"""
import wandb


def init_run(config: dict, job_type: str, run_name: str = None):
    """Start a W&B run tagged with the pipeline stage (job_type), logging
    the FULL resolved config so every run is independently reproducible
    from the dashboard alone, without needing to check git history."""
    return wandb.init(
        project=config["project"]["wandb_project"],
        entity=config["project"].get("wandb_entity"),
        job_type=job_type,
        name=run_name,
        config=config,
    )


def log_dataset_artifact(run, path: str, name: str, description: str):
    artifact = wandb.Artifact(name=name, type="dataset", description=description)
    artifact.add_dir(path)
    run.log_artifact(artifact)


def log_model_artifact(run, path: str, name: str, description: str):
    artifact = wandb.Artifact(name=name, type="model", description=description)
    artifact.add_dir(path)
    run.log_artifact(artifact)


def log_results_table(run, columns: list, rows: list, table_name: str):
    table = wandb.Table(columns=columns, data=rows)
    run.log({table_name: table})