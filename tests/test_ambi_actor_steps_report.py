"""Actor-count specific coverage, trace streaming and paired reporting tests."""
import base64
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from slurm import ambi_actor_steps_report as reporter


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
            "metrics": {k: 10 + seed - 101 + decision - critic for k in reporter.TRAINING_METRICS} if is_critic else {k: seed-101+decision-actor for k in reporter.ACTOR_TRAINING_METRICS}}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(reporter, "source_identity", lambda *args: {"source_sha256": "same"})
    campaign = {"attempt_label": "critic-test", "output_root": str(tmp_path), "rounds": 1,
                "actor_updates": reporter.ACTOR_UPDATES, "critic_updates": 32, "seeds": reporter.SEEDS,
                "controller_seeds": reporter.CONTROLLERS, "cells": [], "worker_receipts": []}
    for count in reporter.ACTOR_UPDATES:
        for controller in reporter.CONTROLLERS:
            cell_id = f"a{count}-s{controller}"; directory = tmp_path / cell_id; directory.mkdir()
            cell = {"cell_id": cell_id, "actor_updates": count, "critic_updates": 32, "rounds": 1, "controller_seed": controller,
                    "seeds": reporter.SEEDS, "bundle": str(directory), "selector": "initialization/inherited", "reused": count == 4}
            params = {"inner_rounds": 1, "inner_rollout_horizon": 1, "inner_rollouts_per_round": 128,
                      "inner_actor_updates_per_round": count, "inner_critic_updates_per_round": 32, "inner_batch_size": 256,
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
                    model = {"inner_actor_optimizer_steps": count, "inner_critic_optimizer_steps": 32,
                             "inner_optimization_model_steps": 128, "inner_actor_q_mean_all_minus_min_all": 2.}
                    if count:
                        model["inner_critic_loss"] = 1.
                    probes = [{"round_index": r, "actor_updates": count*r, "critic_updates": 32*r,
                               "metrics": {"togo_return_gain_vs_outer": {"mean": r, "count": length, "sum": r*length}}} for r in (0,1)]
                    rows.append({"seed": seed, "solver_seed": seed + controller, "return": seed + gain,
                                 "length": length, "terminated": True, "truncated": False, "control_seconds": (count+32)*length*.1,
                                 "togo_probe_seconds": .05*length, "togo_probe_model_steps": 96*length,
                                 "model_metrics": model, "togo_round_summaries": probes})
                    if variant == "inherited":
                        for decision in range(length):
                            events.extend(update_event(seed,decision,step,0,True) for step in range(1,33))
                            events.extend(update_event(seed,decision,32,step,False) for step in range(1,count+1))
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
                        "code": {"runtime": {"torch": "2.3.1"}, "commit": "historical" if count==4 else "new", "dirty": False}}
            hashes = seal_bundle(directory, manifest)
            if cell["reused"]: cell.update(hashes)
            campaign["cells"].append(cell)
    return campaign


def load(campaign):
    return {c["cell_id"]:reporter.load_cell(c) for c in campaign["cells"]}


def test_complete_hierarchical_pairing_and_work(campaign):
    result=reporter.build_report(campaign,load(campaign),resamples=20)
    assert len(result['paired_rows'])==240 and len(result['episode_averages'])==80
    assert len(result['critic_training'])==128 and len(result['actor_training'])==15
    assert sum(r['reused'] for r in result['paired_rows'])==60
    for row in result['summaries']:
        a=row['actor_updates']
        assert row['metrics']['return']['mean']==pytest.approx(115.5+2*a)
        assert row['metrics']['gain_vs_a1']['mean']==pytest.approx(2*(a-1))
        assert row['metrics']['gain_vs_a4']['ci95_low']==pytest.approx(2*(a-4))
        assert row['metrics']['optimizer_updates_per_decision']['mean']==a+32
        assert row['metrics']['critic_updates_per_decision']['mean']==32
        assert row['metrics']['actor_updates_per_episode']['mean']==pytest.approx(a*2.95)
        assert row['metrics']['model_transitions_per_decision']['mean']==128
        probes=[p for p in result['model_probes'] if p['configured_actor_updates']==a]
        assert [(p['actor_updates'],p['critic_updates']) for p in probes]==[(0,0),(a,32)]
    pairs={(r['target_actor_updates'],r['reference_actor_updates']):r for r in result['contrasts']}
    assert set(pairs)=={(1,4),(2,4),(2,1),(4,2),(8,4)}
    assert pairs[8,4]['roles']==['versus_historical_a4','adjacent_actor_dose']
    assert all(r['summary']['mean']==2*(a-b) for (a,b),r in pairs.items())


def test_streamed_critic_actor_step_means_weight_episodes(campaign):
    loaded=load(campaign); cell=loaded['a8-s55']
    actor=cell['rows'][0]['actor_training_steps']
    assert len(actor)==8 and len(cell['rows'][0]['critic_training_steps'])==32
    assert actor[0]['metrics']['actor_loss']=={'count':2,'mean':-.5,'std':.5,'min':-1.,'max':0.}
    result=reporter.build_report(campaign,loaded,resamples=20)
    point=next(r for r in result['actor_training'] if r['configured_actor_updates']==8 and r['actor_updates']==8)
    assert point['metrics']['actor_loss']['mean']==pytest.approx(2.475)
    assert point['metrics']['actor_loss']['episode_count']==20


@pytest.mark.parametrize('mutation,message',[
    ('duplicate','out-of-order'),('missing','Incomplete critic/actor'),('actor_first','out-of-order'),
    ('wrong_counter','Wrong actor update counter'),('bad_metric','Missing/nonfinite'),('bad_hash','checksum mismatch')])
def test_actor_trace_validation(campaign,mutation,message):
    cell=next(c for c in campaign['cells'] if c['actor_updates']==2)
    directory=Path(cell['bundle']); path=directory/'updates.jsonl.gz'
    rows=[json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
    if mutation=='duplicate':rows.insert(33,copy.deepcopy(rows[32]))
    elif mutation=='missing':rows.pop()
    elif mutation=='actor_first':rows[0],rows[32]=rows[32],rows[0]
    elif mutation=='wrong_counter':rows[32]['actor_updates']=4
    elif mutation=='bad_metric':rows[32]['metrics'].pop('actor_grad_norm')
    path.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in rows)+'\n').encode(),mtime=0))
    if mutation!='bad_hash':seal_bundle(directory,reporter.read(directory/'manifest.json'))
    else:path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match=message):reporter.load_cell(cell)


def test_only_actor_count_differs_and_complete_paired_panel_required(campaign):
    loaded=load(campaign);loaded.pop('a1-s55')
    with pytest.raises(ValueError,match='incomplete'):reporter.build_report(campaign,loaded)
    loaded=load(campaign);loaded['a2-s55']['compatibility']['alg_params_except_actor_updates']['inner_critic_updates_per_round']=64
    with pytest.raises(ValueError,match='configuration mismatch'):reporter.build_report(campaign,loaded)
    loaded=load(campaign);loaded['a2-s55']['rows'][0]['solver_seed']+=1
    with pytest.raises(ValueError,match='RNG'):reporter.build_report(campaign,loaded)
    wrong=copy.deepcopy(campaign);wrong['actor_updates'].append(16)
    with pytest.raises(ValueError,match='sweep'):reporter.validate_campaign(wrong)


def test_a4_historical_file_pin(campaign):
    cell=next(c for c in campaign['cells'] if c['actor_updates']==4)
    loaded=reporter.load_cell(cell)
    assert loaded['identity']['seal_sha256']==cell['bundle_seal_sha256']
    assert loaded['identity']['seal_content_sha256']!=cell['bundle_seal_sha256']
    cell['bundle_seal_sha256']='f'*64
    with pytest.raises(ValueError,match='Historical bundle identity'):reporter.load_cell(cell)


class FakeRun:
    def __init__(self):self.logs,self.axes,self.summary,self.finished=[],[],{},[]
    def define_metric(self,*a,**k):self.axes.append((a,k))
    def log(self,row):self.logs.append(row)
    def log_artifact(self,a):self.artifact=a
    def finish(self,**k):self.finished.append(k)
class FakeArtifact:
    def __init__(self,*a,**k):self.files=[]
    def add_file(self,p,**k):self.files.append((p,k))
    def add_dir(self,p,**k):self.files.append((p,k))


def test_html_raw_roundtrip_wandb_actor_axes_and_idempotence(campaign,tmp_path):
    campaign['attempt_label']='</script><script>bad</script>'
    result=reporter.build_report(campaign,load(campaign),resamples=20)
    html=reporter.render_html(result)
    assert campaign['attempt_label'] not in html
    encoded=html.split('<script id="raw-gzip" type="application/octet-stream">')[1].split('</script>')[0]
    assert json.loads(gzip.decompress(base64.b64decode(encoded)))==result
    run=FakeRun();fake=SimpleNamespace(Html=lambda *a,**k:'html',Artifact=FakeArtifact)
    reporter.publish_science(run,fake,result,tmp_path)
    assert [r['compute/actor_updates_per_decision'] for r in run.logs[:4]]==[1,2,4,8]
    assert [r['probes/a8/actor_updates'] for r in run.logs if 'probes/a8/actor_updates' in r]==[0,8]
    assert [r['training/a8/actor_update_index'] for r in run.logs if 'training/a8/actor_update_index' in r]==list(range(1,9))
    assert [r['training/a8/critic_update_index'] for r in run.logs if 'training/a8/critic_update_index' in r]==list(range(1,33))
    assert run.summary['contrasts/a8_minus_a4/mean']==8
    assert run.summary['comparison/historical_rows']==60
    n=len(run.logs);reporter.publish_science(run,fake,result,tmp_path);assert len(run.logs)==n


def test_production_receipt_gate_and_cache(campaign,tmp_path,monkeypatch):
    path=tmp_path/'campaign.json';reporter.write(path,campaign);sha=reporter.file_digest(path)
    calls=[];monkeypatch.setattr(reporter,'load_cell',lambda c:calls.append(c['cell_id']) or {'ok':True})
    loaded={};assert reporter.inspect(campaign,loaded,sha)['complete_cells']==0 and not calls
    for cell in campaign['cells']:
        receipt={k:cell[k] for k in ('cell_id','actor_updates','critic_updates','rounds','reused','seeds')}
        receipt.update(status='complete',campaign_sha256=sha)
        reporter.write(tmp_path/'production'/cell['cell_id']/'merge-completion.json',receipt)
    assert reporter.inspect(campaign,loaded,sha)['complete_cells']==12
    reporter.inspect(campaign,loaded,sha);assert len(calls)==12
    assert reporter.inspect(campaign,{},reporter.digest(campaign))['complete_cells']==0


def test_progress_before_workers_done_no_partial_science_resume(campaign,tmp_path,monkeypatch):
    path=tmp_path/'campaign.json';reporter.write(path,campaign)
    runs,inits=[],[]
    def init(**kwargs):inits.append(kwargs);r=FakeRun();runs.append(r);return r
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init))
    def inspect(*a):
        assert runs[-1].logs[0]['progress/complete_cells']==0
        return {'complete_cells':3,'expected_cells':12,'complete_workers':0,'expected_workers':36,
                'missing':[{'cell_id':'a1-s55','reason':'waiting'}]}
    monkeypatch.setattr(reporter,'inspect',inspect)
    monkeypatch.setattr(reporter,'job_state',lambda *a:{'finished':True})
    args=['--campaign',str(path),'--mode','online','--wandb-run-id','reserved','--watch','--compute-jobs','123']
    assert reporter.main(args)==2 and inits[0]['resume']=='never'
    assert runs[0].finished==[{'exit_code':1}]
    assert not any(any(k.startswith('episodes/') for k in row) for row in runs[0].logs)
    assert reporter.main(args)==2 and inits[1]['resume']=='must'
