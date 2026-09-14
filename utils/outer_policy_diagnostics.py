"""Observational SAC-prior telemetry, independent of learner and simulator RNG."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from RL.tdmpc2_core.outer_policy_diagnostics import policy_diagnostics
from utils.wandb_utils import log_wandb, publish_outer_policy_diagnostics


def _host_packet(value):
    """Transfer a nested packet in one packed copy per device, not per metric."""
    tensors = []
    def collect(v):
        if isinstance(v, torch.Tensor):
            tensors.append(v.detach())
        elif isinstance(v, dict):
            for item in v.values():
                collect(item)
        elif isinstance(v, (tuple, list)):
            for item in v:
                collect(item)
    collect(value)
    converted = {}
    for device in {tensor.device for tensor in tensors}:
        selected = [tensor for tensor in tensors if tensor.device == device]
        packed = torch.cat([tensor.reshape(-1).to(torch.float64) for tensor in selected]).cpu()
        offset = 0
        for tensor in selected:
            size = tensor.numel()
            part = packed[offset:offset + size].reshape(tensor.shape)
            converted[id(tensor)] = part.item() if tensor.ndim == 0 else part.tolist()
            offset += size
    iterator = iter(tensors)
    def rebuild(v):
        if isinstance(v, torch.Tensor):
            return converted[id(next(iterator))]
        if isinstance(v, dict):
            return {key: rebuild(item) for key, item in v.items()}
        if isinstance(v, (tuple, list)):
            return [rebuild(item) for item in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    return rebuild(value)


@contextmanager
def _probe_context(model):
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, training in modes:
            module.training = training


@contextmanager
def _publication_context(cfg):
    """Keep optional publishers from consuming the learner's random streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = set()
    if torch.cuda.is_initialized():
        devices.add(torch.cuda.current_device())
        configured = torch.device(getattr(cfg, "device", "cpu"))
        if configured.type == "cuda" and configured.index is not None:
            devices.add(configured.index)
    try:
        with torch.random.fork_rng(devices=sorted(devices)):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class OuterPolicyDiagnostics:
    """Small fixed-bank probes and an append-only, locally authoritative trace."""

    def __init__(self, cfg, directory, *, state=None):
        self.cfg = cfg
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        count = min(int(cfg.outer_policy_diagnostics_states), int(cfg.seed_steps) + 1)
        self.indices = np.linspace(0, int(cfg.seed_steps), count, dtype=int).tolist()
        self.bank = {}
        generator = torch.Generator(device="cpu").manual_seed(int(cfg.outer_policy_diagnostics_seed))
        self.noise = torch.randn(
            count, int(cfg.outer_policy_diagnostics_samples), int(cfg.action_dim),
            generator=generator,
        )
        self.rows = []
        self.emitted = set()
        self.executed = {}
        self.timing = dict(collection_seconds=0., serialization_seconds=0., publication_seconds=0.)
        self.status = "running"
        if state is not None:
            self.bank = {int(key): torch.as_tensor(value).clone() for key, value in state["bank"].items()}
            self.noise = torch.as_tensor(state["noise"]).clone()
            self.rows = copy.deepcopy(state["rows"])
            self.emitted = {tuple(key) for key in state["emitted"]}
            self.executed = copy.deepcopy(state["executed"])
            self.timing.update(state["timing"])
        trace = self.directory / "events.jsonl"
        if trace.exists() and state is None:
            raise FileExistsError(f"Refusing to overwrite diagnostic trace: {trace}")
        with trace.open("w") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        self._write_manifest()

    def state_dict(self):
        return dict(
            bank={key: value.clone() for key, value in self.bank.items()},
            noise=self.noise.clone(), rows=copy.deepcopy(self.rows),
            emitted=sorted(self.emitted), executed=copy.deepcopy(self.executed),
            timing=dict(self.timing),
        )

    @staticmethod
    def validate_state(state, cfg):
        if not isinstance(state, dict) or set(state) != {"bank", "noise", "rows", "emitted", "executed", "timing"}:
            raise ValueError("Invalid outer policy diagnostic resume fields.")
        count = min(int(cfg.outer_policy_diagnostics_states), int(cfg.seed_steps) + 1)
        noise = state["noise"]
        if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != (
            count, int(cfg.outer_policy_diagnostics_samples), int(cfg.action_dim)
        ) or not torch.isfinite(noise).all():
            raise ValueError("Invalid outer policy diagnostic noise.")
        valid_indices = set(np.linspace(0, int(cfg.seed_steps), count, dtype=int).tolist())
        if not isinstance(state["bank"], dict) or not set(state["bank"]) <= valid_indices:
            raise ValueError("Invalid outer policy diagnostic observation indices.")
        for observation in state["bank"].values():
            if not isinstance(observation, torch.Tensor) or tuple(observation.shape) != tuple(cfg.obs_shape[cfg.obs]):
                raise ValueError("Invalid outer policy diagnostic observation shape.")
        if not isinstance(state["rows"], list) or any(
            row.get("event_index") != index for index, row in enumerate(state["rows"])
        ):
            raise ValueError("Invalid outer policy diagnostic event order.")
        if set(state["timing"]) != {"collection_seconds", "serialization_seconds", "publication_seconds"} or any(
            not np.isfinite(value) or value < 0 for value in state["timing"].values()
        ):
            raise ValueError("Invalid outer policy diagnostic timing.")
        return copy.deepcopy(state)

    def observe(self, observation, env_step):
        started = time.perf_counter()
        completed_bank = False
        if int(env_step) in self.indices and int(env_step) not in self.bank:
            self.bank[int(env_step)] = torch.as_tensor(observation).detach().cpu().clone()
            completed_bank = len(self.bank) == len(self.indices)
        self.timing["collection_seconds"] += time.perf_counter() - started
        if completed_bank:
            # Persist the scientific reference before learning, even if the
            # process later exits without normal finalization.
            self._write_manifest()

    def action(self, action, *, prior):
        started = time.perf_counter()
        source = "executed_prior" if prior else "executed_uniform"
        values = np.asarray(action, dtype=np.float64).reshape(-1)
        stats = self.executed.setdefault(source, dict(count=0, decisions=0, sum=0., sum_sq=0., near=0, exact=0))
        stats["count"] += values.size
        stats["decisions"] += 1
        stats["sum"] += float(values.sum())
        stats["sum_sq"] += float(np.square(values).sum())
        stats["near"] += int((np.abs(values) >= .99).sum())
        stats["exact"] += int((np.abs(values) == 1.).sum())
        self.timing["collection_seconds"] += time.perf_counter() - started

    def _emit(self, packet, *, source, phase, env_step, updates, run):
        started = time.perf_counter()
        packet = _host_packet(packet)
        self.timing["collection_seconds"] += time.perf_counter() - started
        row = dict(schema_version=1, event_index=len(self.rows), source=source, phase=phase,
                   env_step=int(env_step), updates_completed=int(updates), **packet)
        started = time.perf_counter()
        line = json.dumps(row, allow_nan=False)
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
        self.rows.append(row)
        self.timing["serialization_seconds"] += time.perf_counter() - started
        payload = {f"outer_diag/{source}/{key}": value for key, value in packet.get("metrics", {}).items()}
        payload.update({"outer_diag/updates_completed": int(updates),
                        "outer_diag/phase": phase, "outer_diag/source": source})
        started = time.perf_counter()
        try:
            if run is not None:
                with _publication_context(self.cfg):
                    log_wandb(run, payload, step=int(env_step))
        finally:
            self.timing["publication_seconds"] += time.perf_counter() - started

    def learner(self, packet, *, env_step, updates, phase, run):
        self.timing["collection_seconds"] += float(packet.get("diagnostic_collection_seconds", 0.))
        self._emit(packet, source="learner", phase=phase, env_step=env_step, updates=updates, run=run)

    def flush_actions(self, *, env_step, updates, phase, run):
        for source, stats in list(self.executed.items()):
            if not stats["count"]:
                continue
            n = stats["count"]
            mean = stats["sum"] / n
            metrics = dict(action_mean=mean, action_std=max(0., stats["sum_sq"] / n - mean * mean) ** .5,
                           action_near_saturation_fraction=stats["near"] / n,
                           action_exact_saturation_fraction=stats["exact"] / n,
                           coordinate_count=n, decision_count=stats["decisions"])
            before = len(self.rows)
            try:
                self._emit(dict(metrics=metrics, measurement_timing="collection_window"),
                           source=source, phase=phase, env_step=env_step, updates=updates, run=run)
            finally:
                # _emit commits locally before publishing. Retrying a failed
                # remote write must not count these executed actions twice.
                if len(self.rows) > before:
                    self.executed.pop(source)
        self.executed.clear()

    def probe(self, agent, *, env_step, updates, phase, run, observation=None):
        source = "initial_observation" if observation is not None else "reference_bank"
        key = (source, int(updates))
        if key in self.emitted:
            return
        if observation is None:
            if not self.bank:
                return
            observations = torch.stack([self.bank[key] for key in sorted(self.bank)])
        else:
            observations = torch.as_tensor(observation).detach().cpu().unsqueeze(0)
        started = time.perf_counter()
        samples = int(self.cfg.outer_policy_diagnostics_samples)
        model = agent.model
        with _probe_context(model):
            z = model.encode(observations.to(agent.device))
            action, info = model.pi(
                z.repeat_interleave(samples, dim=0),
                noise=self.noise[:len(observations)].reshape(-1, int(self.cfg.action_dim)).to(agent.device),
            )
            packet = policy_diagnostics(info, action, lower=float(self.cfg.log_std_min),
                                        upper=float(self.cfg.log_std_max), rho=float(self.cfg.rho))
            # Parameter histograms count state/action coordinates once; sampling
            # the same Gaussian repeatedly does not add parameter observations.
            for histogram in packet["histograms"].values():
                histogram["counts"] = histogram["counts"] // samples
                histogram["count"] //= samples
            # There is one latent depth here; its entropy is a joint-action estimate.
            entropy = info["entropy"].detach().mean()
            alpha = agent.alpha.detach().reshape(())
            packet["metrics"].update(entropy_joint_nats=entropy, alpha=alpha,
                                      entropy_bonus=alpha * entropy)
            if agent.target_entropy is not None:
                packet["metrics"].update(entropy_target=float(agent.target_entropy),
                                          entropy_shortfall=float(agent.target_entropy) - entropy)
            packet.update(measurement_timing="post_update" if updates else "before_learning",
                          actor_updates_before=int(updates), actor_updates_after=int(updates),
                          observation_count=len(observations), samples_per_state=samples,
                          parameter_coordinate_count=len(observations) * int(self.cfg.action_dim),
                          sampled_coordinate_count=len(observations) * samples * int(self.cfg.action_dim),
                          encoder="current", noise_seed=int(self.cfg.outer_policy_diagnostics_seed))
            packet = _host_packet(packet)
        self.timing["collection_seconds"] += time.perf_counter() - started
        before = len(self.rows)
        try:
            self._emit(packet, source=source, phase=phase, env_step=env_step, updates=updates, run=run)
        finally:
            if len(self.rows) > before:
                self.emitted.add(key)
        self.flush_actions(env_step=env_step, updates=updates, phase=phase, run=run)

    def _write_manifest(self):
        started = time.perf_counter()
        bank = {str(key): value.tolist() for key, value in sorted(self.bank.items())}
        provenance = dict(indices=self.indices, observations=bank, noise=self.noise.tolist())
        encoded = json.dumps(provenance, sort_keys=True).encode()
        (self.directory / "reference.json").write_bytes(encoded)
        # The resolved critic target determines the meaning of decoded Q.
        # Preserve the historical soft-Q label when older configs omit it.
        q_units = (
            "decoded predicted reward-only return, not measured return"
            if getattr(self.cfg, "outer_critic_target", "entropy_augmented") == "reward_only"
            else "decoded soft Q, not reward-only return"
        )
        manifest = dict(schema="ambi-outer-policy-diagnostics", version=1, status=self.status,
                        row_count=len(self.rows), reference_sha256=hashlib.sha256(encoded).hexdigest(),
                        config=vars(self.cfg), timing=dict(self.timing),
                        initialization="native_random", histograms="24 equal-width log-std bins",
                        entropy_units="joint nats", q_units=q_units,
                        timing_note="collection includes device transfer; publication measures enqueue calls")
        temp = self.directory / "manifest.json.tmp"
        temp.write_text(json.dumps(manifest, default=str, indent=2, allow_nan=False) + "\n")
        temp.replace(self.directory / "manifest.json")
        self.timing["serialization_seconds"] += time.perf_counter() - started

    def finish(self, agent, *, env_step, updates, run, failed=False, publish_artifact=True):
        # Final scientific rows and their manifest become durable before any
        # remote call. A failed W&B connection must not leave a running manifest
        # or prevent the last executed-action window from reaching the bundle.
        first_final_row = len(self.rows)
        try:
            if not failed:
                self.probe(agent, env_step=env_step, updates=updates, phase="final", run=None)
            self.flush_actions(env_step=env_step, updates=updates, phase="final", run=None)
        except BaseException as primary_error:
            self.status = "incomplete"
            try:
                self._write_manifest()
            except BaseException as cleanup_error:
                from utils.cleanup import add_cleanup_notes
                add_cleanup_notes(primary_error, (cleanup_error,),
                                  prefix="Additional diagnostic manifest cleanup failure")
            raise
        self.status = "incomplete" if failed else "complete"
        self._write_manifest()
        if run is not None:
            # Replay only rows produced locally above, without appending them a
            # second time. log_wandb retains the event adapter/resume journal.
            with _publication_context(self.cfg):
                for row in self.rows[first_final_row:]:
                    payload = {f"outer_diag/{row['source']}/{key}": value
                               for key, value in row.get("metrics", {}).items()}
                    payload.update({"outer_diag/updates_completed": row["updates_completed"],
                                    "outer_diag/phase": row["phase"], "outer_diag/source": row["source"]})
                    started = time.perf_counter()
                    try:
                        log_wandb(run, payload, step=row["env_step"])
                    finally:
                        self.timing["publication_seconds"] += time.perf_counter() - started
                log_wandb(run, {"outer_diag/updates_completed": int(updates),
                                **{f"outer_diag/time/{key}": value for key, value in self.timing.items()}}, step=env_step)
            if publish_artifact:
                self.attach(run, env_step=env_step, updates=updates)

    def attach(self, run, *, env_step, updates):
        """Attach the finalized bundle, returning artifact enqueue seconds.

        Resumable callers invoke this after their final checkpoint commit. They
        receive the timing as a return value rather than a new, uncommitted
        journal event. Ordinary runs also record the timing in W&B history.
        An adjacent ``<bundle-name>_publication.json`` receipt records enqueue
        timing/status without changing the immutable bundle after capture.
        """
        from utils.wandb_resume import CheckpointedWandbRun

        receipt_path = self.directory.with_name(self.directory.name + "_publication.json")

        def receipt(status, elapsed, error=None):
            manifest = self.directory / "manifest.json"
            payload = dict(
                schema="ambi-outer-policy-diagnostics-publication", version=1,
                status=status, enqueue_seconds=elapsed, env_step=int(env_step),
                updates_completed=int(updates), run_id=getattr(run, "id", None),
                timing=dict(self.timing),
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
                note="queued means the SDK accepted the artifact; this is not a remote verification",
            )
            if error is not None:
                payload["error"] = str(error)
            temporary = receipt_path.with_name(receipt_path.name + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            temporary.replace(receipt_path)

        if run is None:
            receipt("disabled", 0.0)
            return 0.0
        with _publication_context(self.cfg):
            started = time.perf_counter()
            try:
                publish_outer_policy_diagnostics(run, self.directory)
            except BaseException as primary_error:
                elapsed = time.perf_counter() - started
                try:
                    receipt("failed", elapsed, primary_error)
                except BaseException as cleanup_error:
                    from utils.cleanup import add_cleanup_notes
                    add_cleanup_notes(primary_error, (cleanup_error,),
                                      prefix="Additional diagnostic publication receipt failure")
                raise
            elapsed = time.perf_counter() - started
            receipt("queued", elapsed)
            if not isinstance(run, CheckpointedWandbRun):
                log_wandb(run, {"outer_diag/updates_completed": int(updates),
                                "outer_diag/time/artifact_enqueue_seconds": elapsed}, step=env_step)
            else:
                print(f"Outer diagnostic artifact queued in {elapsed:.3f}s; receipt: {receipt_path}", flush=True)
        return elapsed
