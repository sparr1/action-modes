"""Publication is complete, explicit, content-verified and idempotent."""
import copy
import gzip
import json
from pathlib import Path

import pytest

import evaluate_mppi_saturation as diagnostic
import publish_mppi_saturation as publication


def _json(path, value):
    path.write_text(json.dumps(value))


def _gzip(path, rows):
    with gzip.open(path, "wt") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _seal(directory):
    _json(directory / "checksums.json", {name: publication._hash(directory / name) for name in publication.FILES})


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    roots = [dict(episode_id=f"seed-{seed}", root_id=f"seed-{seed}-decision-{decision}",
                  seed=seed, decision_index=decision) for seed in range(101, 121) for decision in [0, 100, 200, 300, 400]]
    protocol = dict(seeds=list(range(101, 121)), controller_seed=55, action_rule="tanh_mean",
                    decisions=[0, 100, 200, 300, 400], max_steps=500, science="a" * 64)
    bank = dict(roots=roots, protocol=protocol, checkpoint_sha256="b" * 64)
    bank["id"] = publication._canonical_hash(bank)
    _json(directory / "root-bank.json", bank)
    _json(directory / "matrix.json", {"shared_alg_params": {"inner_rounds": 4}})
    _json(directory / "checkpoint.metadata.json", {"checkpoint": {"step": 200000}})
    rows = []
    for operator, rounds, distributions in [("mppi", [1, 2, 3, 4], publication.MPPI_DISTRIBUTIONS),
            ("sac", [0, 1, 2, 4], ["policy_samples", "policy_mean"])]:
        for iteration in rounds:
            for distribution in distributions:
                rows.append(dict(**roots[0], solver_repeat=0, operator=operator, iteration=iteration,
                    distribution=distribution, actor_updates=iteration * 4, critic_updates=iteration * 32,
                    **{metric: .5 for metric in publication.METRICS}))
    _gzip(directory / "measurements.jsonl.gz", rows)
    _gzip(directory / "mppi-populations.jsonl.gz", [dict(**roots[0], solver_repeat=0, iteration=j,
        actions=[[[0., 0.]] * 128], values=[[0.]] * 128, elite_indices=list(range(16)), elite_weights=[[1 / 16]] * 16)
        for j in range(1, 5)])
    record = dict(schema_version=1, kind="mppi_sac_action_saturation", status="complete", smoke=True,
        outer_state_unchanged=True, roots=1, solver_repetitions=1, sac_rounds=[0, 1, 2, 4], sac_policy_samples=1024,
        bootstrap_resamples=2000, bootstrap_seed=0, root_bank_sha256=publication._hash(directory / "root-bank.json"),
        matrix_sha256=publication._hash(directory / "matrix.json"), root_bank_id=bank["id"],
        checkpoint_sha256="b" * 64, root_protocol=protocol, science="a" * 64,
        diagnostic_source_sha256="c" * 64, sac_resolved_config={"inner_rounds": 4}, action_rules={"optimized_mean": "mean"},
        mppi=dict(horizon=1, iterations=4, num_samples=128, num_elites=16, num_pi_trajs=6, temperature=.5,
                  min_std=.05, max_std=2., eval_mode=True), summary=diagnostic.aggregate_rows(rows),
        elapsed_seconds=1.5, timing={"mppi_seconds": .25, "inner_sac_seconds": 1.})
    _json(directory / "results.json", record)
    diagnostic.write_html(directory / "report.html", record)
    _seal(directory)
    return directory


class FakeRun:
    def __init__(self, fail=False):
        self.history, self.axes, self.artifacts, self.finished = [], [], [], []
        self.summary = {}
        self.fail = fail
    def define_metric(self, *args, **kwargs):
        self.axes.append((args, kwargs))
    def log(self, row):
        self.history.append(row)
        if self.fail:
            raise RuntimeError("simulated network failure")
    def log_artifact(self, artifact):
        self.artifacts.append(artifact)
    def finish(self, **kwargs):
        self.finished.append(kwargs)


class FakeArtifact:
    def __init__(self, *args, **kwargs):
        self.files = {}
        self.metadata = kwargs["metadata"]
    def add_file(self, path, name):
        self.files[name] = publication._hash(path)


class FakeWandb:
    Artifact = FakeArtifact
    def __init__(self, fail=False):
        self.calls = []
        self.run = FakeRun(fail)
    def init(self, **kwargs):
        self.calls.append(kwargs)
        return self.run


def _publish(bundle, fake, **kwargs):
    return publication.publish(bundle, attempt_label="explicit-attempt", run_id="unique-test-run", mode="disabled",
                               allow_smoke=True, wandb_module=fake, **kwargs)


def test_publication_preserves_artifacts_round_rows_and_source_identity(bundle):
    fake = FakeWandb()
    record, checksums = publication.read_completed(bundle, allow_smoke=True)
    receipt = _publish(bundle, fake)
    assert receipt["status"] == "complete" and receipt["history_rows"] == 5
    assert [row["saturation/round"] for row in fake.run.history] == [0, 1, 2, 3, 4]
    assert fake.run.history == publication.history_rows(record)
    assert "saturation/sac/policy_samples/exact_boundary_fraction/mean" not in fake.run.history[3]
    assert fake.calls[0]["id"] == "unique-test-run" and fake.calls[0]["resume"] == "never"
    assert fake.calls[0]["config"]["root_protocol"] == record["root_protocol"]
    assert fake.run.artifacts[0].files == {**checksums, "checksums.json": publication._hash(bundle / "checksums.json")}
    assert fake.run.summary["final/mppi/optimized_mean/exact_boundary_fraction/mean"] == .5
    assert fake.run.finished == [{}]
    assert _publish(bundle, fake) == receipt and len(fake.calls) == 1


def test_failed_publication_is_never_blindly_resumed(bundle):
    fake = FakeWandb(fail=True)
    with pytest.raises(RuntimeError, match="network"):
        _publish(bundle, fake)
    assert fake.run.finished == [{"exit_code": 1}]
    assert json.loads((bundle / "publication.json").read_text())["status"] == "uncertain"
    with pytest.raises(ValueError, match="uncertain"):
        _publish(bundle, fake)
    assert len(fake.calls) == 1


def test_smoke_requires_explicit_publication_permission(bundle):
    fake = FakeWandb()
    with pytest.raises(ValueError, match="allow-smoke"):
        publication.publish(bundle, attempt_label="attempt", run_id="unique-run", wandb_module=fake)
    assert fake.calls == [] and not (bundle / "publication.json").exists()


@pytest.mark.parametrize("mutation", ["checksum", "missing_row", "duplicate_row", "bad_summary", "bad_ci", "missing_population", "frozen", "protocol"])
def test_invalid_results_fail_before_wandb_or_journal(bundle, mutation):
    fake = FakeWandb()
    if mutation == "checksum":
        (bundle / "report.html").write_text("corrupt")
    elif mutation in ("missing_row", "duplicate_row"):
        rows = list(publication._rows(bundle / "measurements.jsonl.gz"))
        if mutation == "missing_row":
            rows.pop()
        else:
            rows[-1] = copy.deepcopy(rows[0])
        _gzip(bundle / "measurements.jsonl.gz", rows)
        _seal(bundle)
    elif mutation == "missing_population":
        rows = list(publication._rows(bundle / "mppi-populations.jsonl.gz"))
        _gzip(bundle / "mppi-populations.jsonl.gz", rows[:-1])
        _seal(bundle)
    else:
        record = json.loads((bundle / "results.json").read_text())
        if mutation == "bad_summary":
            record["summary"][0]["metrics"]["exact_boundary_fraction"]["mean"] = .9
        elif mutation == "bad_ci":
            record["summary"][0]["metrics"]["exact_boundary_fraction"]["ci95"] = [0., 1.]
        elif mutation == "frozen":
            record["outer_state_unchanged"] = False
        else:
            record["root_protocol"]["controller_seed"] = 56
        _json(bundle / "results.json", record)
        _seal(bundle)
    with pytest.raises(ValueError):
        _publish(bundle, fake)
    assert fake.calls == [] and not (bundle / "publication.json").exists()


def test_changed_attempt_cannot_reuse_completed_run(bundle):
    fake = FakeWandb()
    _publish(bundle, fake)
    with pytest.raises(ValueError, match="target differs"):
        publication.publish(bundle, attempt_label="different", run_id="unique-test-run", mode="disabled",
                            allow_smoke=True, wandb_module=fake)
    assert len(fake.calls) == 1


def test_oscar_launchers_pin_source_and_separate_gpu_from_online_publisher():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "slurm/run_mppi_saturation_oscar.sbatch").read_text()
    publisher = (root / "slurm/run_mppi_saturation_publish_oscar.sbatch").read_text()
    for text in (worker, publisher):
        assert "EXPECTED_ACTION_MODES_SHA" in text and "git status --porcelain" in text
        assert "environments/dmcontrol/.venv/bin/python" in text
    assert "WANDB_MODE=offline" in worker and "publish_mppi_saturation.py" not in worker
    assert "--smoke" in worker and "AMBI_MPPI_SMOKE" in worker
    assert "WANDB_MODE=online" in publisher and "AMBI_MPPI_RUN_ID" in publisher
