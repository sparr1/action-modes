"""Exercise the probe from a portable checkpoint to inspectable local artifacts."""

import copy
import json

import numpy as np
import pytest
import torch

import probe_pave
from tests.test_ambi_root_local_sac import _model_from_params, _tiny_params
from utils.pave_probe_data import PaveProbeBank, save_probe_bank
from utils.pave_probe_report import recommend_scales, distribution


@pytest.fixture
def checkpoint_and_bank(tmp_path):
    params = _tiny_params(aux_return_mode="sac", critic_coef=.1, aux_return_critic_coef=.1)
    model = _model_from_params(params)
    try:
        with torch.no_grad():
            for heads in (model.agent.model._Qs, model.agent.model._aux_return_Qs):
                for head in heads:
                    head[-1].weight.normal_(std=.03)
        checkpoint = tmp_path / "model.pt"
        torch.save(model.agent.checkpoint_state(), checkpoint)
    finally:
        model.env.close()
        model.close()
    env_signature = dict(id="Pendulum-v1", params={"max_episode_steps": 5}, wrappers=[])
    metadata = {
        "schema_version": 1,
        "checkpoint": {"kind": "step", "step": 25000, "episode": 50, "best_score": None, "best_window": 100},
        "trial_run_params": {"alg": "AMBITDMPC2/AMBITDMPC2", "env": "Pendulum-v1",
                             "seed": 13, "device": "cpu", "total_steps": 10, "alg_params": params},
        "experiment_params": {"env_params": env_signature["params"]},
    }
    (tmp_path / "model.pt.metadata.json").write_text(json.dumps(metadata))
    observations = np.random.default_rng(123).normal(size=(9, 3)).astype(np.float32)
    bank = PaveProbeBank(
        observations=observations[:-1], next_observations=observations[1:],
        actions=np.full((8, 1), .2, np.float32), rewards=np.ones((8, 1), np.float32),
        terminated=np.zeros((8, 1), bool), truncated=np.array([[False]] * 7 + [[True]]),
        episode_ids=np.zeros(8, np.int64), episode_steps=np.arange(8),
        provenance={"source_kind": "training_replay", "env_signature": env_signature},
    )
    path = tmp_path / "bank.npz"
    save_probe_bank(path, bank)
    return checkpoint, path


def _arguments(tmp_path, checkpoint_and_bank, *extras):
    checkpoint, bank = checkpoint_and_bank
    return ["--checkpoint", str(checkpoint), "--bank", str(bank),
            "--output-dir", str(tmp_path / "probe"), "--batches", "2", "--warmup-batches", "0",
            "--scales", "0", ".1", "1", *extras]


def test_end_to_end_frozen_multi_checkpoint_probe(tmp_path, checkpoint_and_bank, monkeypatch):
    checkpoint, bank = checkpoint_and_bank
    before = checkpoint.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Frozen probe attempted an optimizer step")

    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    assert probe_pave.main(_arguments(tmp_path, checkpoint_and_bank,
                                    "--checkpoint", str(checkpoint))) == 0
    result = json.loads((tmp_path / "probe/results.json").read_text())
    assert result["complete"] and len(result["checkpoints"]) == 2
    assert result["protocol"]["optimizer_steps"] == 0
    assert checkpoint.read_bytes() == before
    first, second = result["checkpoints"]
    assert first["summary"]["batch_count"] == 2
    assert set(first["summary"]["critics"]) == {"primary", "auxiliary"}
    for row in result["checkpoints"]:
        assert row["frozen_state"]["unchanged"]
        assert row["resolved_cfg"]["seed"] == 13
        assert row["bank"]["provenance"]["source_kind"] == "training_replay"
        assert row["bank"]["sha256"] == probe_pave._file_sha256(bank)
    # Exact values reproduce; timings necessarily do not.
    assert first["batches"][0]["critics"] == second["batches"][0]["critics"]
    report = (tmp_path / "probe/report.md").read_text()
    assert "stronger_curvature" in report and "Adam" in report
    assert not (tmp_path / "probe/failure.json").exists()
    assert probe_pave.main(_arguments(tmp_path, checkpoint_and_bank)) == 1
    assert json.loads((tmp_path / "probe/results.json").read_text()) == result


def test_explicit_stochastic_prior_collection(tmp_path, checkpoint_and_bank):
    checkpoint, _ = checkpoint_and_bank
    args = ["--checkpoint", str(checkpoint), "--collect-episodes", "2", "--max-steps", "3",
            "--output-dir", str(tmp_path / "collected"), "--batches", "1", "--warmup-batches", "0",
            "--critics", "auxiliary"]
    assert probe_pave.main(args) == 0
    result = json.loads((tmp_path / "collected/results.json").read_text())
    row = result["checkpoints"][0]
    assert row["bank"]["provenance"]["source_kind"] == "stochastic_prior_proxy"
    assert row["bank"]["transition_count"] == 6
    assert row["bank"]["provenance"]["source_checkpoint_sha256"] == row["checkpoint"]["sha256"]
    assert (tmp_path / "collected/checkpoint-0-bank.npz").is_file()


@pytest.mark.parametrize("extra", [
    ["--warmup-batches", "-1"], ["--seed", "-1"], ["--scales", ".1", ".1"],
    ["--critics", "primary", "primary"],
])
def test_reject_invalid_protocol_before_output(tmp_path, checkpoint_and_bank, extra):
    assert probe_pave.main(_arguments(tmp_path, checkpoint_and_bank, *extra)) == 1
    assert not (tmp_path / "probe").exists()


def test_mismatched_environment_bank_is_rejected(tmp_path, checkpoint_and_bank):
    checkpoint, bank_path = checkpoint_and_bank
    from utils.pave_probe_data import load_probe_bank
    bank = load_probe_bank(bank_path)
    bank.provenance["env_signature"]["params"] = {"max_episode_steps": 6}
    other_path = tmp_path / "other.npz"
    save_probe_bank(other_path, bank)
    args = _arguments(tmp_path, (checkpoint, other_path))
    assert probe_pave.main(args) == 1
    assert not (tmp_path / "probe/results.json").exists()
    assert json.loads((tmp_path / "probe/failure.json").read_text())["complete"] is False


def _summary_row(scale, ratios, clip=1.):
    return {"variant": "reference", "scope": "primary", "scale": scale,
            "critics": {"primary": {"ratio_to_td": distribution(ratios)}},
            "clip_multiplier_relative_to_baseline": distribution([clip])}


def _checkpoint_summary(candidates):
    return {"summary": {"candidates": candidates, "critics": {
        "primary": {"term_gradient_ratios_to_td": {
            term: distribution([1.]) for term in ("mpr", "vfc", "curvature")
        }}
    }}}


def test_scale_screen_checks_each_checkpoint_and_missing_ratios():
    first = _checkpoint_summary([_summary_row(.1, [.15, .2]), _summary_row(1., [1.5, 2.])])
    second = copy.deepcopy(first)
    selected = recommend_scales([first, second])["selections"]["primary"]
    assert selected["scale"] == .1
    second["summary"]["candidates"][0] = _summary_row(.1, [.15, None])
    assert recommend_scales([first, second])["selections"]["primary"]["scale"] is None
    second["summary"]["candidates"][0] = _summary_row(.1, [.15, .2], clip=.8)
    assert recommend_scales([first, second])["selections"]["primary"]["scale"] is None
    second["summary"]["candidates"][0] = _summary_row(.1, [.01, .02])
    assert recommend_scales([first, second])["selections"]["primary"]["scale"] is None
    second = copy.deepcopy(first)
    second["summary"]["critics"]["primary"]["term_gradient_ratios_to_td"]["mpr"] = distribution([100.])
    assert recommend_scales([first, second])["selections"]["primary"]["scale"] is None


def test_scale_screen_does_not_recalibrate_stronger_curvature():
    reference = _summary_row(.1, [.2])
    stronger = copy.deepcopy(reference)
    stronger["variant"] = "stronger_curvature"
    stronger["critics"]["primary"]["ratio_to_td"] = distribution([4.])
    result = recommend_scales([_checkpoint_summary([reference, stronger])])
    selection = result["selections"]["primary"]
    assert selection["scale"] == .1
    assert selection["stronger_curvature_at_same_scale"][0]["critics"]["primary"]["ratio_to_td"]["median"] == 4.
