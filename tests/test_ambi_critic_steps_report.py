"""Critic-count specific coverage, trace streaming and paired reporting tests."""
import base64
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from slurm import ambi_critic_steps_report as reporter


def seal_bundle(directory, manifest):
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {"kind": "ambi_episode_seed_merge", "expected_seeds": reporter.SEEDS, "sources": []}
    manifest["seed_shard_merge"] = receipt
    reporter.write(directory / "manifest.json", manifest)
    reporter.write(directory / "seed-shard-merge.json", receipt)
    names = ["manifest.json", "seed-shard-merge.json", *[p for r in manifest["runs"] for p in r.get("trace_files", [])]]
    seal = {"kind": "ambi_episode_seed_shard", "files": {name: reporter.file_digest(directory / name) for name in names}}
    seal["sha256"] = reporter.digest(seal)
    reporter.write(directory / "seed-shard-checksums.json", seal)
    return {"bundle_manifest_sha256": seal["files"]["manifest.json"],
            "bundle_seal_sha256": reporter.file_digest(directory / "seed-shard-checksums.json")}


def update_event(seed, decision, critic, actor, is_critic):
    return {"episode_id": f"seed-{seed}", "decision_index": decision, "phase": "update", "round_index": 1,
            "measurement": "pre_update_minibatch", "updated_critic": is_critic, "updated_actor": not is_critic,
            "updated_temperature": False, "critic_updates": critic, "actor_updates": actor,
            "metrics": {k: 10 + seed - 101 + decision - critic for k in reporter.TRAINING_METRICS} if is_critic else {"actor_loss": -1}}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "source_identity", lambda *args: {"source_sha256": "same"})
    campaign = {"attempt_label": "critic-test", "output_root": str(tmp_path), "rounds": 1,
                "critic_updates": reporter.CRITIC_UPDATES, "seeds": reporter.SEEDS,
                "controller_seeds": reporter.CONTROLLERS, "cells": [], "worker_receipts": []}
    for count in reporter.CRITIC_UPDATES:
        for controller in reporter.CONTROLLERS:
            cell_id = f"c{count}-s{controller}"; directory = tmp_path / cell_id; directory.mkdir()
            cell = {"cell_id": cell_id, "critic_updates": count, "rounds": 1, "controller_seed": controller,
                    "seeds": reporter.SEEDS, "bundle": str(directory), "selector": "initialization/inherited", "reused": count == 32}
            params = {"inner_rounds": 1, "inner_rollout_horizon": 1, "inner_rollouts_per_round": 128,
                      "inner_actor_updates_per_round": 4, "inner_critic_updates_per_round": count, "inner_batch_size": 256,
                      "inner_temperature": 0., "inner_actor_entropy_mode": "tdmpc2_scaled",
                      "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
                      "inner_temperature_mode": "fixed", "inner_temperature_initialization": "fixed",
                      "inner_execution_action": "mean", "inner_behavior_action": "policy_sample",
                      "inner_finite_horizon": True, "inner_sac_critic_target": "reward_only", "inner_actor_lr": .0003}
            runs = []
            for variant in ("prior", "inherited"):
                rows, events = [], []
                for seed in reporter.SEEDS:
                    length = 2 if seed == 101 else 3
                    gain = 0 if variant == "prior" else 5 + count * 2 + controller - 56
                    model = {"inner_actor_optimizer_steps": 4, "inner_critic_optimizer_steps": count,
                             "inner_optimization_model_steps": 128, "inner_actor_q_mean_all_minus_min_all": 2.}
                    if count:
                        model["inner_critic_loss"] = 1.
                    probes = [{"round_index": r, "actor_updates": 4*r, "critic_updates": count*r,
                               "metrics": {"togo_return_gain_vs_outer": {"mean": r, "count": length, "sum": r*length}}} for r in (0,1)]
                    rows.append({"seed": seed, "solver_seed": seed + controller, "return": seed + gain,
                                 "length": length, "terminated": True, "truncated": False, "control_seconds": (count+4)*length*.1,
                                 "togo_probe_seconds": .05*length, "togo_probe_model_steps": 96*length,
                                 "model_metrics": model, "togo_round_summaries": probes})
                    if variant == "inherited":
                        for decision in range(length):
                            events.extend(update_event(seed,decision,step,0,True) for step in range(1,count+1))
                            events.extend(update_event(seed,decision,count,step,False) for step in range(1,5))
                trace_files = []
                if events:
                    name = "updates.jsonl.gz"
                    (directory/name).write_bytes(gzip.compress(('\n'.join(json.dumps(e) for e in events)+'\n').encode(),mtime=0))
                    trace_files.append(name)
                config = {"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": copy.deepcopy(params)}
                result = {"environment_seeds": reporter.SEEDS, "controller_seed": controller,
                          "outer_state_unchanged": True, "outer_updates_before": 20, "outer_updates_after": 20,
                          "alg_params": copy.deepcopy(params), "episodes": rows,
                          "togo_return_probe": {"rollouts": 32, "horizon": 1, "tail_actor": "outer",
                            "tail_critic": "outer_online", "tail_q_reduction": "mean_pair", "entropy_bonus": False}}
                runs.append({"selector": "initialization/"+variant, "status": "complete", "kind": "episodes",
                             "config": config, "config_hash": reporter.digest(config), "episodes": copy.deepcopy(rows),
                             "result": result, "trace_files": trace_files})
            manifest = {"status": "complete", "runs": runs,
                        "checkpoint": {"sha256": reporter.CHECKPOINT, "source_run": "rwgao_b-brown-university/ambi/mey3rxj8",
                                       "source_run_verified": False, "metadata": {"checkpoint": {"step": 200000}}},
                        "protocol": {"action_rule": "tanh_mean", "controller_seed": controller, "max_steps": 500},
                        "code": {"runtime": {"torch": "2.3.1"}, "commit": "historical" if count==32 else "new", "dirty": False}}
            hashes = seal_bundle(directory, manifest)
            if cell["reused"]: cell.update(hashes)
            campaign["cells"].append(cell)
    return campaign


def load(campaign):
    return {c["cell_id"]:reporter.load_cell(c) for c in campaign["cells"]}


def test_pairing_compute_and_c0_structural_absence(campaign):
    result = reporter.build_report(campaign,load(campaign),resamples=20)
    assert len(result['paired_rows']) == 420 and len(result['episode_averages']) == 140
    assert len(result['critic_training']) == sum(reporter.CRITIC_UPDATES)
    assert sum(r['reused'] for r in result['paired_rows']) == 60
    for row in result['summaries']:
        c = row['critic_updates']
        assert row['metrics']['return']['mean'] == pytest.approx(115.5+2*c)
        assert row['metrics']['gain_vs_c0']['mean'] == pytest.approx(2*c)
        assert row['metrics']['gain_vs_c32']['ci95_low'] == pytest.approx(2*(c-32))
        assert row['metrics']['optimizer_updates_per_decision']['mean'] == c+4
        assert row['metrics']['actor_updates_per_episode']['mean'] == pytest.approx(4*2.95)
        assert row['metrics']['model_transitions_per_decision']['mean'] == 128
        assert ('inner_critic_loss' in row['model_metrics']) == (c>0)
    c0 = [r for r in result['model_probes'] if r['configured_critic_updates']==0]
    assert [(r['actor_updates'],r['critic_updates']) for r in c0] == [(0,0),(4,0)]
    assert not any(r['configured_critic_updates']==0 for r in result['critic_training'])
    assert all(r['critic_training_steps']==[] for r in result['paired_rows'] if r['critic_updates']==0)


def test_streamed_step_means_weight_decisions_then_episodes(campaign):
    loaded = load(campaign)
    cell = loaded['c4-s55']
    first = cell['rows'][0]['critic_training_steps'][0]['metrics']['critic_loss']
    assert first['count'] == 2 and first['mean'] == 9.5 and first['std'] == .5
    result = reporter.build_report(campaign,loaded,resamples=20)
    point = next(r for r in result['critic_training'] if r['configured_critic_updates']==4 and r['critic_updates']==1)
    # Episode101 has2decisions, others3; give each episode equal weight.
    assert point['metrics']['critic_loss']['mean'] == pytest.approx(19.475)
    assert point['metrics']['critic_loss']['episode_count'] == 20
    assert len(cell['identity']['training_trace_sources']) == 1


@pytest.mark.parametrize('mutation,message', [
    ('duplicate','Duplicate or out-of-order'),('missing','Incomplete critic/actor'),
    ('actor_first','Duplicate or out-of-order'),('critic_at_c0','Wrong critic update counter'),
    ('bad_metric','Missing/nonfinite'),('bad_hash','checksum mismatch')])
def test_training_trace_validation(campaign,mutation,message):
    cell = next(c for c in campaign['cells'] if c['critic_updates']==(0 if mutation=='critic_at_c0' else 4))
    directory = Path(cell['bundle']); path=directory/'updates.jsonl.gz'
    events = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    if mutation=='duplicate': events.insert(1,copy.deepcopy(events[0]))
    elif mutation=='missing': events.pop()
    elif mutation=='actor_first': events[0],events[4]=events[4],events[0]
    elif mutation=='critic_at_c0': events.insert(0,update_event(101,0,1,0,True))
    elif mutation=='bad_metric': events[0]['metrics'].pop('critic_loss')
    path.write_bytes(gzip.compress(('\n'.join(json.dumps(e) for e in events)+'\n').encode(),mtime=0))
    if mutation!='bad_hash': seal_bundle(directory,reporter.read(directory/'manifest.json'))
    else: path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match=message): reporter.load_cell(cell)


def test_only_critic_count_can_differ_and_missing_panel_rejected(campaign):
    values=load(campaign)
    values.pop('c0-s55')
    with pytest.raises(ValueError,match='incomplete'): reporter.build_report(campaign,values)
    values=load(campaign)
    values['c1-s55']['compatibility']['alg_params_except_critic_updates']['inner_actor_lr']=.1
    with pytest.raises(ValueError,match='configuration mismatch'): reporter.build_report(campaign,values)
    values=load(campaign); values['c1-s55']['rows'][0]['solver_seed']+=1
    with pytest.raises(ValueError,match='RNG'): reporter.build_report(campaign,values)


def test_c32_seal_file_pin_and_actual_work_required(campaign):
    cell=next(c for c in campaign['cells'] if c['critic_updates']==32)
    assert cell['bundle_seal_sha256'] != reporter.read(Path(cell['bundle'])/'seed-shard-checksums.json')['sha256']
    loaded=reporter.load_cell(cell)
    assert loaded['identity']['seal_sha256']==cell['bundle_seal_sha256']
    assert loaded['identity']['seal_content_sha256']!=cell['bundle_seal_sha256']
    cell['bundle_seal_sha256']='f'*64
    with pytest.raises(ValueError,match='Historical bundle identity'): reporter.load_cell(cell)
    cell=campaign['cells'][0]; directory=Path(cell['bundle']); manifest=reporter.read(directory/'manifest.json')
    manifest['runs'][1]['result']['episodes'][0]['model_metrics']['inner_actor_optimizer_steps']=0
    seal_bundle(directory,manifest)
    with pytest.raises(ValueError,match='realized actor'): reporter.load_cell(cell)


class FakeRun:
    def __init__(self): self.logs,self.axes,self.summary,self.finished=[],[],{},[]
    def define_metric(self,*a,**k):self.axes.append((a,k))
    def log(self,row):self.logs.append(row)
    def log_artifact(self,artifact):self.artifact=artifact
    def finish(self,**k):self.finished.append(k)
class FakeArtifact:
    def __init__(self,*a,**k):self.files=[]
    def add_file(self,p,**k):self.files.append((p,k))
    def add_dir(self,p,**k):self.files.append((p,k))


def test_html_roundtrip_and_distinct_training_axes(campaign,tmp_path):
    campaign['attempt_label']='</script><script>bad</script>'
    result=reporter.build_report(campaign,load(campaign),resamples=20)
    html=reporter.render_html(result)
    assert campaign['attempt_label'] not in html
    encoded=html.split('<script id="raw-gzip" type="application/octet-stream">')[1].split('</script>')[0]
    assert json.loads(gzip.decompress(base64.b64decode(encoded)))==result
    assert 'not a post-update held-out error' in html
    run=FakeRun();fake=SimpleNamespace(Html=lambda *a,**k:'html',Artifact=FakeArtifact)
    reporter.publish_science(run,fake,result,tmp_path)
    assert [r['compute/critic_updates_per_decision'] for r in run.logs[:7]] == reporter.CRITIC_UPDATES
    assert not any(any(k.startswith('critic_training/c0/') for k in row) for row in run.logs)
    assert any('critic_training/c64/td_error_abs_mean/mean' in row for row in run.logs)
    assert not any('model_transitions' in args[0] for args,_ in run.axes)
    count=len(run.logs);reporter.publish_science(run,fake,result,tmp_path);assert len(run.logs)==count


def test_merge_receipt_blocks_before_loading_and_complete_receipt_caches(campaign,tmp_path,monkeypatch):
    path=tmp_path/'campaign.json';reporter.write(path,campaign);sha=reporter.file_digest(path)
    calls=[]
    monkeypatch.setattr(reporter,'load_cell',lambda cell: calls.append(cell['cell_id']) or {'ok':True})
    loaded={};assert reporter.inspect(campaign,loaded,sha)['complete_cells']==0 and not calls
    for cell in campaign['cells']:
        receipt={k:cell[k] for k in ('cell_id','critic_updates','reused','seeds')}
        receipt.update(status='complete',campaign_sha256=sha)
        reporter.write(tmp_path/'production'/cell['cell_id']/'merge-completion.json',receipt)
    assert reporter.inspect(campaign,loaded,sha)['complete_cells']==21
    reporter.inspect(campaign,loaded,sha);assert len(calls)==21
    assert reporter.inspect(campaign,{},'bad')['complete_cells']==0


def test_progress_new_id_then_receipt_bound_resume_and_no_partial_science(campaign,tmp_path,monkeypatch):
    path=tmp_path/'campaign.json';reporter.write(path,campaign)
    runs,inits=[],[]
    def init(**kwargs):inits.append(kwargs);r=FakeRun();runs.append(r);return r
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init))
    def inspect(*a):
        assert runs[-1].logs[0]['progress/complete_cells']==0
        return {'complete_cells':3,'expected_cells':21,'complete_workers':0,'expected_workers':21,
                'missing':[{'cell_id':'c0-s55','reason':'waiting'}]}
    monkeypatch.setattr(reporter,'inspect',inspect)
    monkeypatch.setattr(reporter,'job_state',lambda *a:{'finished':True})
    args=['--campaign',str(path),'--mode','online','--wandb-run-id','reserved','--watch','--compute-jobs','123']
    assert reporter.main(args)==2 and inits[0]['resume']=='never'
    assert runs[0].finished==[{'exit_code':1}]
    assert not any(any(k.startswith('episodes/') for k in row) for row in runs[0].logs)
    assert reporter.main(args)==2 and inits[1]['resume']=='must'
