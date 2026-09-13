"""C64 J2's paired comparison and cumulative multi-round trace contract."""
import base64
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import pytest
from slurm import ambi_c64_rounds_report as reporter


def seal_bundle(directory, manifest):
    receipt = {"kind": "ambi_episode_seed_merge", "expected_seeds": reporter.SEEDS, "sources": []}
    manifest["seed_shard_merge"] = receipt
    reporter.write(directory / "manifest.json", manifest)
    reporter.write(directory / "seed-shard-merge.json", receipt)
    names = ["manifest.json", "seed-shard-merge.json", *[p for r in manifest["runs"] for p in r.get("trace_files", [])]]
    seal = {"kind": "ambi_episode_seed_shard", "files": {n: reporter.file_digest(directory / n) for n in names}}
    seal["sha256"] = reporter.digest(seal)
    reporter.write(directory / "seed-shard-checksums.json", seal)
    return {"bundle_manifest_sha256": seal["files"]["manifest.json"],
            "bundle_seal_sha256": reporter.file_digest(directory / "seed-shard-checksums.json")}


def event(seed, decision, round_index, critic, actor, is_critic):
    return {"episode_id": f"seed-{seed}", "decision_index": decision, "phase": "update", "round_index": round_index,
            "measurement": "pre_update_minibatch", "updated_critic": is_critic, "updated_actor": not is_critic,
            "updated_temperature": False, "critic_updates": critic, "actor_updates": actor,
            "metrics": {k: 10 + seed - 101 + decision - critic for k in reporter.TRAINING_METRICS} if is_critic else {"actor_loss": -1}}


def events_for(seed, decision, count, rounds):
    rows = []
    for r in range(1, rounds + 1):
        rows.extend(event(seed, decision, r, step, 4*(r-1), True) for step in range(count*(r-1)+1, count*r+1))
        rows.extend(event(seed, decision, r, count*r, step, False) for step in range(4*(r-1)+1, 4*r+1))
    return rows


def write_trace(path, events):
    path.write_bytes(gzip.compress(('\n'.join(json.dumps(e) for e in events)+'\n').encode(), mtime=0))


@pytest.fixture(scope='module')
def panel(tmp_path_factory):
    root = tmp_path_factory.mktemp('c64-round-panel')
    campaign = {"attempt_label": "c64-j2-test", "output_root": str(root), "rounds": [1,2],
                "critic_updates": [32,64], "seeds": reporter.SEEDS, "controller_seeds": reporter.CONTROLLERS,
                "cells": [], "worker_receipts": []}
    for count in reporter.CRITIC_UPDATES:
        for rounds in reporter.ROUNDS:
            for controller in reporter.CONTROLLERS:
                cell_id = f"c{count}-j{rounds}-s{controller}"
                directory = root / cell_id; directory.mkdir()
                reused = count == 32 or rounds == 1
                cell = {"cell_id": cell_id, "critic_updates": count, "rounds": rounds, "controller_seed": controller,
                        "seeds": reporter.SEEDS, "bundle": str(directory), "selector": "initialization/inherited", "reused": reused}
                params = {"inner_rounds": rounds, "inner_rollout_horizon": 1, "inner_rollouts_per_round": 128,
                          "inner_actor_updates_per_round": 4, "inner_critic_updates_per_round": count, "inner_batch_size": 256,
                          "inner_temperature": 0., "inner_actor_entropy_mode": "tdmpc2_scaled",
                          "inner_actor_initialization": "prior", "inner_critic_initialization": "prior",
                          "inner_temperature_mode": "fixed", "inner_temperature_initialization": "fixed",
                          "inner_execution_action": "mean", "inner_behavior_action": "policy_sample",
                          "inner_finite_horizon": True, "inner_sac_critic_target": "reward_only", "inner_actor_lr": .0003}
                runs = []
                for variant in ('prior','inherited'):
                    rows, events = [], []
                    for seed in reporter.SEEDS:
                        length = 2 if seed == 101 else 3
                        gain = 0 if variant == 'prior' else rounds*(10+count/32)+controller-56
                        model = {"inner_actor_optimizer_steps": 4*rounds, "inner_critic_optimizer_steps": count*rounds,
                                 "inner_optimization_model_steps": 128*rounds, "inner_critic_loss": 1.}
                        probes = [{"round_index": r, "actor_updates": 4*r, "critic_updates": count*r,
                                   "metrics": {"togo_return_gain_vs_outer": {"mean": r, "count": length, "sum": r*length}}}
                                  for r in range(rounds+1)]
                        rows.append({"seed": seed, "solver_seed": seed+controller, "return": seed+gain,
                                     "length": length, "terminated": True, "truncated": False,
                                     "control_seconds": (count+4)*rounds*length*.1,
                                     "togo_probe_seconds": .05*length, "togo_probe_model_steps": 96*length,
                                     "model_metrics": model, "togo_round_summaries": probes})
                        if variant == 'inherited':
                            for decision in range(length):
                                events.extend(events_for(seed,decision,count,rounds))
                    trace_files=[]
                    if events:
                        write_trace(directory/'updates.jsonl.gz',events); trace_files=['updates.jsonl.gz']
                    config={"alg": "AMBITDMPC2/AMBITDMPC2", "alg_params": copy.deepcopy(params)}
                    result={"environment_seeds":reporter.SEEDS,"controller_seed":controller,"outer_state_unchanged":True,
                            "outer_updates_before":20,"outer_updates_after":20,"alg_params":copy.deepcopy(params),"episodes":rows,
                            "togo_return_probe":{"rollouts":32,"horizon":1,"tail_actor":"outer","tail_critic":"outer_online",
                                                 "tail_q_reduction":"mean_pair","entropy_bonus":False}}
                    runs.append({"selector":"initialization/"+variant,"status":"complete","kind":"episodes",
                                 "config":config,"config_hash":reporter.digest(config),"episodes":copy.deepcopy(rows),
                                 "result":result,"trace_files":trace_files})
                manifest={"status":"complete","runs":runs,
                          "checkpoint":{"sha256":reporter.CHECKPOINT,"source_run":"rwgao_b-brown-university/ambi/mey3rxj8",
                                        "source_run_verified":False,"metadata":{"checkpoint":{"step":200000}}},
                          "protocol":{"action_rule":"tanh_mean","controller_seed":controller,"max_steps":500},
                          "code":{"runtime":{"torch":"2.3.1"},"commit":"historical" if reused else "new","dirty":False}}
                hashes=seal_bundle(directory,manifest)
                if reused: cell.update(hashes)
                campaign['cells'].append(cell)
    with patch.object(reporter,'source_identity',lambda *args:{'source_sha256':'same'}):
        loaded={c['cell_id']:reporter.load_cell(c) for c in campaign['cells']}
    return campaign,loaded


def test_complete_pairing_rounds_compute_and_interaction(panel):
    campaign, loaded = panel
    result=reporter.build_report(campaign,loaded,resamples=20)
    assert len(result['paired_rows'])==240 and len(result['episode_averages'])==80
    assert sum(r['reused'] for r in result['paired_rows'])==180
    assert len(result['critic_training'])==288
    for row in result['summaries']:
        c,j=row['critic_updates'],row['rounds']
        assert row['metrics']['return']['mean']==pytest.approx(110.5+j*(10+c/32))
        assert row['metrics']['gain_vs_j1']['mean']==pytest.approx((j-1)*(10+c/32))
        assert row['metrics']['gain_vs_c32']['ci95_low']==pytest.approx(j*(c-32)/32)
        assert row['metrics']['actor_updates_per_decision']['mean']==4*j
        assert row['metrics']['critic_updates_per_decision']['mean']==c*j
        assert row['metrics']['model_transitions_per_decision']['mean']==128*j
        assert row['metrics']['actor_updates_per_episode']['mean']==pytest.approx(4*j*2.95)
    contrasts={r['label']:r for r in result['contrasts']}
    assert len(contrasts)==5
    for label,value in [('C64: J2 minus J1',12),('J2: C64 minus C32',2),('C64 minus C32: J2 minus J1',1)]:
        assert contrasts[label]['summary']['mean']==value
        assert contrasts[label]['summary']['ci95_low']==pytest.approx(value)
        assert len(contrasts[label]['episode_differences'])==20
    probes=[p for p in result['model_probes'] if p['configured_critic_updates']==64 and p['configured_rounds']==2]
    assert [(p['round_index'],p['actor_updates'],p['critic_updates']) for p in probes]==[(0,0,0),(1,4,64),(2,8,128)]


def test_streamed_training_round_boundary_and_episode_weight(panel):
    campaign,loaded=panel
    cell=loaded['c64-j2-s55']; points=cell['rows'][0]['critic_training_steps']
    assert len(points)==128
    assert (points[64]['round_index'],points[64]['critic_update_in_round'],points[64]['actor_updates'],points[64]['critic_updates'])==(2,1,4,65)
    assert points[0]['metrics']['critic_loss']=={'count':2,'mean':9.5,'std':.5,'min':9.,'max':10.}
    result=reporter.build_report(campaign,loaded,resamples=20)
    point=next(p for p in result['critic_training'] if p['configured_critic_updates']==64 and p['configured_rounds']==2 and p['critic_updates']==65)
    # Each episode's decision average gets equal weight despite lengths2and3.
    assert point['metrics']['critic_loss']['mean']==pytest.approx(19.475-64)
    assert point['metrics']['critic_loss']['episode_count']==20
    assert cell['identity']['training_trace_sources'][0]['sha256']==reporter.file_digest(Path(cell['cell']['bundle'])/'updates.jsonl.gz')


@pytest.mark.parametrize('mutation,message',[
    ('skipped_actor','out-of-order'),('duplicate_critic','out-of-order'),('reset_counter','counter'),
    ('wrong_round','counter'),('missing_final','Incomplete'),('bad_hash','checksum'),('bad_metric','Missing/nonfinite')])
def test_multiround_trace_rejects_invalid_sequences(tmp_path,mutation,message):
    events=events_for(101,0,2,2)
    if mutation=='skipped_actor': events.pop(5)
    elif mutation=='duplicate_critic': events.insert(7,copy.deepcopy(events[6]))
    elif mutation=='reset_counter': events[6]['critic_updates']=1
    elif mutation=='wrong_round': events[6]['round_index']=1
    elif mutation=='missing_final': events.pop()
    elif mutation=='bad_metric': events[6]['metrics']['critic_loss']=None
    path=tmp_path/'trace.jsonl.gz';write_trace(path,events)
    seal={'files':{'trace.jsonl.gz':reporter.file_digest(path)}}
    if mutation=='bad_hash': seal['files']['trace.jsonl.gz']='x'*64
    cell={'bundle':str(tmp_path),'critic_updates':2,'rounds':2}
    with pytest.raises(ValueError,match=message):
        reporter.training_trace_summaries(cell,{'trace_files':['trace.jsonl.gz']},seal,[{'seed':101,'length':1}])


def test_rejects_j4_incomplete_unpaired_and_other_scientific_changes(panel):
    campaign,loaded=panel
    wrong=copy.deepcopy(campaign);wrong['rounds']=[1,2,4]
    with pytest.raises(ValueError,match='sweep'): reporter.validate_campaign(wrong)
    missing=dict(loaded);missing.pop('c64-j2-s55')
    with pytest.raises(ValueError,match='incomplete'):reporter.build_report(campaign,missing)
    wrong=copy.deepcopy(loaded);wrong['c64-j2-s55']['compatibility']['alg_params_except_critic_updates_and_rounds']['inner_actor_lr']=.1
    with pytest.raises(ValueError,match='configuration mismatch'):reporter.build_report(campaign,wrong)
    wrong=copy.deepcopy(loaded);wrong['c64-j2-s55']['rows'][0]['solver_seed']+=1
    with pytest.raises(ValueError,match='RNG'):reporter.build_report(campaign,wrong)
    wrong=copy.deepcopy(loaded);wrong['c64-j2-s55']['prior'][0]['return']+=1
    with pytest.raises(ValueError,match='frozen-prior'):reporter.build_report(campaign,wrong)


def test_historical_seal_file_namespace(panel):
    _,loaded=panel
    cell=loaded['c32-j2-s55']
    assert cell['identity']['seal_sha256']==cell['cell']['bundle_seal_sha256']
    assert cell['identity']['seal_content_sha256']!=cell['cell']['bundle_seal_sha256']


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


def test_html_roundtrip_and_multiround_wandb_axes(panel,tmp_path):
    campaign,loaded=panel;campaign=copy.deepcopy(campaign)
    campaign['attempt_label']='</script><script>bad</script>'
    report=reporter.build_report(campaign,loaded,resamples=20)
    html=reporter.render_html(report)
    assert campaign['attempt_label'] not in html and '<option selected>4</option>' not in html
    encoded=html.split('<script id="raw-gzip" type="application/octet-stream">')[1].split('</script>')[0]
    # JSON roundtrip normalizes tuples in contrast component vectors.
    assert json.loads(gzip.decompress(base64.b64decode(encoded)))==json.loads(json.dumps(report))
    assert 'not a post-update held-out error' in html
    run=FakeRun();fake=SimpleNamespace(Html=lambda *a,**k:'html',Artifact=FakeArtifact)
    reporter.publish_science(run,fake,report,tmp_path)
    assert [r.get('compute/c64/rounds') for r in run.logs[:4]]==[None,None,1.,2.]
    assert [r['probes/c64/j2/actor_updates'] for r in run.logs if 'probes/c64/j2/actor_updates' in r]==[0,4,8]
    assert [r['training/c64/j2/critic_update_index'] for r in run.logs if 'training/c64/j2/critic_update_index' in r]==list(range(1,129))
    assert not any('/j4/' in key for row in run.logs for key in row)
    assert run.summary['comparison/historical_rows']==180
    count=len(run.logs);reporter.publish_science(run,fake,report,tmp_path);assert len(run.logs)==count


def test_receipt_requires_round_and_raw_campaign_sha_and_caches(panel,tmp_path,monkeypatch):
    campaign=copy.deepcopy(panel[0]);campaign['output_root']=str(tmp_path)
    path=tmp_path/'campaign.json';reporter.write(path,campaign);sha=reporter.file_digest(path)
    calls=[];monkeypatch.setattr(reporter,'load_cell',lambda c:calls.append(c['cell_id']) or {'ok':True})
    loaded={};assert reporter.inspect(campaign,loaded,sha)['complete_cells']==0 and not calls
    for c in campaign['cells']:
        receipt={k:c[k] for k in ('cell_id','critic_updates','rounds','reused','seeds')}
        receipt.update(status='complete',campaign_sha256=sha)
        reporter.write(tmp_path/'production'/c['cell_id']/'merge-completion.json',receipt)
    assert reporter.inspect(campaign,loaded,sha)['complete_cells']==12
    reporter.inspect(campaign,loaded,sha);assert len(calls)==12
    assert reporter.inspect(campaign,{},reporter.digest(campaign))['complete_cells']==0
    path=tmp_path/'production'/'c64-j2-s55'/'merge-completion.json'
    receipt=reporter.read(path);receipt['rounds']=1;reporter.write(path,receipt)
    assert reporter.inspect(campaign,{},sha)['complete_cells']==11


def test_immediate_progress_no_partial_science_and_receipt_bound_resume(panel,tmp_path,monkeypatch):
    campaign=copy.deepcopy(panel[0]);campaign['output_root']=str(tmp_path)
    path=tmp_path/'campaign.json';reporter.write(path,campaign)
    runs,inits=[],[]
    def init(**kwargs):inits.append(kwargs);r=FakeRun();runs.append(r);return r
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(init=init))
    def inspect(*a):
        assert runs[-1].logs[0]['progress/complete_cells']==0
        return {'complete_cells':9,'expected_cells':12,'complete_workers':0,'expected_workers':12,
                'missing':[{'cell_id':'c64-j2-s55','reason':'waiting'}]}
    monkeypatch.setattr(reporter,'inspect',inspect)
    monkeypatch.setattr(reporter,'job_state',lambda *a:{'finished':True})
    args=['--campaign',str(path),'--mode','online','--wandb-run-id','reserved','--watch','--compute-jobs','123']
    assert reporter.main(args)==2 and inits[0]['resume']=='never'
    assert runs[0].finished==[{'exit_code':1}]
    assert not any(any(k.startswith('episodes/') for k in row) for row in runs[0].logs)
    assert reporter.main(args)==2 and inits[1]['resume']=='must'
