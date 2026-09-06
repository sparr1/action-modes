import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import publish_tdmpc2_mppi_eval as publisher


SOURCE = "rwgao_b-brown-university/ambi/xq3zva9u"
CAMPAIGN = "prior-mppi-test"


def write_result(root, step=100000, shift=0):
    directory = root / f"step_{step}"
    directory.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    config_name = "tdmpc2_humanoid_walk_state_prior_only_checkpoint_bank_1p5m.json"
    metadata = {
        "checkpoint": {"step": step},
        "trial_run_params": json.loads((repository / "configs/dmcontrol/algs" / config_name).read_text()),
        "experiment_params": json.loads((repository / "configs/dmcontrol/experiments" / config_name).read_text()),
    }
    metadata_path = root / f"original_{step}.metadata.json"
    # Deliberately noncanonical formatting: copying must preserve the pinned byte hash.
    metadata_path.write_text(json.dumps(metadata, indent=4) + "\n\n")
    episodes = []
    for index, seed in enumerate(range(101, 106)):
        arms = {}
        for controller, gain in (("policy_prior_mean", 0), ("native_mppi", 2)):
            arms[controller] = {"return": index + gain + shift, "length": 500,
                                "steps": [{}] * 500, "terminated": False,
                                "truncated": True, "capped": False, "seconds": 1.5}
        episodes.append({"environment_seed": seed, "return_delta": 2, **arms})
    result = {
        "schema_version": 1, "algorithm": "TDMPC2/TDMPC2Baseline", "environment": "DMControl-v0",
        "checkpoint_sha256": f"{step:064x}", "checkpoint_metadata": {"step": step},
        "configuration_source": str(metadata_path), "planner": copy.deepcopy(publisher.PLANNER),
        "frozen_state": {"unchanged": True, "model_digest_before": "abc", "model_digest_after": "abc",
                         "num_updates_before": 100, "num_updates_after": 100},
        "protocol": {"controllers": ["policy_prior_mean", "native_mppi"], "environment_seed_first": 101,
                     "environment_seed_last": 105, "max_steps": 500, "controller_seed_base": 12345},
        "episodes": episodes,
        "summary": {"paired_episodes": 5, "policy_prior_return_mean": 2 + shift,
                    "native_mppi_return_mean": 4 + shift, "paired_return_delta_mean": 2},
    }
    path = directory / "paired.json"
    path.write_text(json.dumps(result))
    provenance = {"source_run": SOURCE, "campaign": CAMPAIGN, "code_sha": "a" * 40,
                  "code_tree": "b" * 40, "checkpoint_sha256": result["checkpoint_sha256"],
                  "metadata_sha256": publisher._hash(metadata_path)}
    (directory / "provenance.json").write_text(json.dumps(provenance))
    return path


def change(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


class FakeWandb:
    def __init__(self):
        self.runs = []
        self.plots = []
        self.artifacts = []
        self.plot = SimpleNamespace(line_series=self.line_series)

    def init(self, **kwargs):
        parent = self

        class Run:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def log(self, data):
                self.logged.append(data)

            def log_artifact(self, artifact):
                parent.artifacts.append(artifact)

        run = Run()
        run.options, run.summary, run.logged = kwargs, {}, []
        self.runs.append(run)
        return run

    def line_series(self, **kwargs):
        self.plots.append(kwargs)
        return kwargs

    def Table(self, **kwargs):
        return kwargs

    def Artifact(self, *args, **kwargs):
        artifact = SimpleNamespace(files=[], metadata=kwargs["metadata"])
        artifact.add_file = lambda path, name: artifact.files.append((path, name))
        return artifact


def test_refresh_adds_new_checkpoints_in_step_order_and_preserves_missing_values(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    later = write_result(tmp_path, 200000, shift=10)
    sdk = FakeWandb()
    first = publisher.publish(later, source_run=SOURCE, campaign=CAMPAIGN, expected_max_step=400000, wandb_module=sdk)
    assert sdk.plots[0]["ys"][0] == [None, None, 12, None, None, None, None]
    earlier = write_result(tmp_path)
    second = publisher.publish(earlier, source_run=SOURCE, campaign=CAMPAIGN, expected_max_step=400000, wandb_module=sdk)
    assert sdk.plots[2]["xs"] == list(publisher.STEPS[:7])
    assert sdk.plots[2]["ys"] == [[2, None, 12, None, None, None, None],
                                  [4, None, 14, None, None, None, None]]
    assert sdk.plots[2]["keys"] == ["Policy prior mean", "Paper MPPI (H3, J8)"]
    assert sdk.plots[3]["ys"] == [[2, None, 2, None, None, None, None]]
    assert first["campaign_run_id"] == second["campaign_run_id"]
    assert first["checkpoint_run_id"] != second["checkpoint_run_id"]
    assert sdk.runs[-1].summary["completed_checkpoint_steps"] == [100000, 200000]
    assert sdk.runs[-1].summary["status"] == "partial"
    assert {name for _, name in sdk.artifacts[-1].files} == {
        "paired.json", "checkpoint.metadata.json", "provenance.json"}
    assert sdk.runs[-2].summary["runtime/mppi_seconds"] == 7.5


def test_disabled_smoke_preserves_sidecar_hash_and_is_portable(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    path = write_result(tmp_path)
    record = publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN, wandb_module=object())
    assert record["status"] == "validated_upload_disabled"
    original = tmp_path / "original_100000.metadata.json"
    copied = path.parent / "checkpoint.metadata.json"
    assert original.read_bytes() == copied.read_bytes()
    original.unlink()
    assert publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN)["result_sha256"] == record["result_sha256"]


@pytest.mark.parametrize("mutate", [
    lambda d: d["frozen_state"].update(unchanged=False),
    lambda d: d["frozen_state"].update(num_updates_after=101),
    lambda d: d["checkpoint_metadata"].update(step=150000),
    lambda d: d.update(checkpoint_sha256="f" * 64),
    lambda d: d["planner"].update(effective_iterations=10),
    lambda d: d["summary"].update(policy_prior_return_mean=99),
    lambda d: d["episodes"][0].update(return_delta=3),
    lambda d: d["episodes"][0]["native_mppi"].update({"return": float("nan")}),
    lambda d: d["episodes"].pop(),
    lambda d: d["episodes"][0]["native_mppi"].update(truncated=False),
])
def test_rejects_invalid_or_incomplete_result(tmp_path, mutate):
    path = write_result(tmp_path)
    change(path, mutate)
    with pytest.raises(ValueError):
        publisher.load_result(path, SOURCE, CAMPAIGN)


@pytest.mark.parametrize("key,value", [("source_run", "foreign/project/run"), ("campaign", "foreign"),
                                        ("code_sha", "c" * 40)])
def test_rejects_foreign_source_or_code_in_sibling(tmp_path, key, value):
    write_result(tmp_path)
    later = write_result(tmp_path, 150000)
    change(later.parent / "provenance.json", lambda d: d.update({key: value}))
    with pytest.raises(ValueError):
        publisher.campaign_data(tmp_path, SOURCE, CAMPAIGN)


def test_rejects_different_controller_rng_protocol(tmp_path):
    write_result(tmp_path)
    later = write_result(tmp_path, 150000)
    change(later, lambda d: d["protocol"].update(controller_seed_base=999))
    with pytest.raises(ValueError, match="protocols"):
        publisher.campaign_data(tmp_path, SOURCE, CAMPAIGN)


def test_rejects_republication_of_replaced_result(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    path = write_result(tmp_path)
    publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN)
    change(path, lambda d: d.update(extra="changed result"))
    with pytest.raises(ValueError, match="different result"):
        publisher.publish(path, source_run=SOURCE, campaign=CAMPAIGN)


def test_complete_campaign_has_all_actual_points_through_1p5m(tmp_path):
    for index, step in enumerate(reversed(publisher.STEPS)):
        write_result(tmp_path, step, shift=step / 100000)
    loaded, curves = publisher.campaign_data(tmp_path, SOURCE, CAMPAIGN)
    assert len(loaded) == 29
    assert curves[0] == [2 + step / 100000 for step in publisher.STEPS]
    assert curves[2] == [2] * 29


def test_extending_campaign_preserves_ids_and_updates_expected_grid(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    sdk = FakeWandb()
    earlier = write_result(tmp_path, 400000)
    first = publisher.publish(earlier, source_run=SOURCE, campaign=CAMPAIGN,
                              expected_max_step=400000, wandb_module=sdk)
    later = write_result(tmp_path, 1150000)
    second = publisher.publish(later, source_run=SOURCE, campaign=CAMPAIGN,
                               expected_max_step=1150000, wandb_module=sdk)
    assert first["campaign_run_id"] == second["campaign_run_id"]
    assert sdk.runs[-1].options["allow_val_change"] is True
    assert sdk.runs[-1].options["config"]["expected_checkpoint_steps"] == list(range(100000, 1150001, 50000))
    assert sdk.runs[-1].summary["completed_checkpoint_steps"] == [400000, 1150000]
    assert sdk.plots[-2]["ys"][0][-1] == 2
    with pytest.raises(ValueError, match="maximum"):
        publisher.campaign_data(tmp_path, SOURCE, CAMPAIGN, expected_max_step=400000)


@pytest.fixture
def source_revisions(tmp_path):
    """Real Git objects verify source identity, without relying on fake hashes."""
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repository, stderr=subprocess.PIPE).decode().strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    for name in publisher.EVALUATION_SOURCE_PATHS:
        path = repository / name
        if "." not in path.name:
            path /= "example.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original source\n")

    def commit():
        git("add", ".")
        git("commit", "-m", "fixture")
        return git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")

    original = commit()
    (repository / "publish_tdmpc2_mppi_eval.py").write_text("publication change\n")
    publication = commit()
    (repository / "RL/example.py").write_text("changed scientific source\n")
    scientific = commit()
    return repository, original, publication, scientific


@pytest.mark.parametrize("mutation", ["publication", "scientific", "forged_tree", "forged_fingerprint"])
def test_mixed_commits_require_git_verified_identical_scientific_source(tmp_path, source_revisions, monkeypatch, mutation):
    repository, original, publication, scientific = source_revisions
    fingerprint = publisher.evaluation_source_fingerprint
    monkeypatch.setattr(publisher, "evaluation_source_fingerprint",
                        lambda sha, tree: fingerprint(sha, tree, repository))
    first = write_result(tmp_path / "results", 400000)
    later = write_result(tmp_path / "results", 450000)
    change(first.parent / "provenance.json", lambda d: d.update(code_sha=original[0], code_tree=original[1]))
    selected = scientific if mutation == "scientific" else publication
    change(later.parent / "provenance.json", lambda d: d.update(
        code_sha=selected[0], code_tree="0" * 40 if mutation == "forged_tree" else selected[1],
        evaluation_source_sha256="f" * 64 if mutation == "forged_fingerprint"
        else fingerprint(*selected, repository)))
    if mutation == "publication":
        loaded, curves = publisher.campaign_data(tmp_path / "results", SOURCE, CAMPAIGN, expected_max_step=450000)
        assert len(loaded) == 2 and curves[0][-2:] == [2, 2]
        assert loaded[0]["provenance"]["code_sha"] == original[0]
        assert loaded[1]["provenance"]["code_sha"] == publication[0]
    else:
        with pytest.raises(ValueError):
            publisher.campaign_data(tmp_path / "results", SOURCE, CAMPAIGN)


def test_installed_wandb_table_preserves_null_gaps():
    import wandb
    table = wandb.Table(columns=["step", "lineKey", "lineVal"],
                        data=[[100000, "prior", 1], [150000, "prior", None], [200000, "prior", 3]])
    assert table.data[1] == [150000, "prior", None]
