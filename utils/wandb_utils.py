"""Optional W&B helpers used by AMBI-native algorithms.

Importing this module does not import wandb. W&B is imported only when a run is
explicitly enabled with `wandb: true` in an algorithm config.
"""

from __future__ import annotations


DEFAULT_WANDB_ENTITY = "rwgao_b-brown-university"
DEFAULT_WANDB_PROJECT = "ambi"


def wandb_enabled(params: dict | None) -> bool:
    return bool(params and params.get("wandb", False))


def init_wandb(params: dict | None, *, default_project: str, run_name: str | None, config: dict | None = None):
    if not wandb_enabled(params):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb=True but the wandb package is not installed. Run `pip install wandb` or set wandb=false.") from exc

    project = params.get("wandb_project", default_project or DEFAULT_WANDB_PROJECT)
    entity = params.get("wandb_entity", DEFAULT_WANDB_ENTITY)
    name = params.get("wandb_run_name", run_name)
    mode = params.get("wandb_mode", None)
    tags = params.get("wandb_tags", None)

    kwargs = {"project": project, "entity": entity, "name": name, "config": config or {}}
    if mode:
        kwargs["mode"] = mode
    if tags:
        kwargs["tags"] = tags

    run = wandb.init(**kwargs)
    wandb.define_metric("env_step")
    wandb.define_metric("train/*", step_metric="env_step")
    wandb.define_metric("rollout/*", step_metric="env_step")
    wandb.define_metric("episode/*", step_metric="env_step")
    wandb.define_metric("time/*", step_metric="env_step")
    return run


def log_wandb(run, payload: dict, *, step: int) -> None:
    if run is None:
        return
    payload = dict(payload)
    payload.setdefault("env_step", int(step))
    run.log(payload, step=int(step))


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()
