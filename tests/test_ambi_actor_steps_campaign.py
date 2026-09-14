"""Actor-step coverage, immutable reuse, and paired actor optimization prefixes."""
import copy
import inspect
import json
from pathlib import Path

import pytest

from slurm import ambi_actor_steps_campaign as launch


@pytest.fixture
def campaign(tmp_path):
    cells = []
    for count in launch.ACTOR_UPDATES:
        for seed in launch.CONTROLLERS:
            cell_id = f"a{count}-s{seed}"
            reused = count == 4
            output = tmp_path / ("historical" if reused else "production") / cell_id
            matrix = "round_scaling_j1_h1_200k.json" if reused else f"actor_steps_a{count}_j1_h1_200k.json"
            width = 5
            cells.append(dict(cell_id=cell_id, actor_updates=count, critic_updates=32, rounds=1, controller_seed=seed,
                matrix=f"configs/research/{matrix}", selector=launch.SELECTOR,
                seeds=launch.SEEDS, seed_shards=[launch.SEEDS[i:i+width] for i in range(0,20,width)],
                reused=reused, bundle=str(output / "bundle"), model_series=str(output / "model-series"),
                identity_spec=str(tmp_path / "specs" / cell_id / "initialization__inherited.json"),
                **({"bundle_manifest_sha256":"a"*64,"bundle_seal_sha256":"b"*64} if reused else {})))
    data = dict(schema_version=1, attempt_label="test-actor-steps", output_root=str(tmp_path),
                inventory=str(tmp_path / "inventory.json"), actor_updates=launch.ACTOR_UPDATES, critic_updates=32, rounds=1,
                seeds=launch.SEEDS, controller_seeds=launch.CONTROLLERS, cells=cells)
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(data))
    return path


def mock_runtime(monkeypatch, tmp_path):
    calls = []
    def run(args, **kwargs):
        command = list(map(str, args))
        calls.append(command)
        if command[:2] == ["merge_ambi_seed_shards.py", "episodes"]:
            Path(command[command.index("--output")+1]).mkdir(parents=True)
    monkeypatch.setattr(launch, "_run", run)
    monkeypatch.setattr(launch, "load_inventory", lambda *args: [dict(step=200000,sha256="c"*64)])
    monkeypatch.setattr(launch, "verify_checkpoint", lambda row: tmp_path / "checkpoint")
    return calls


def test_matrices_change_only_actor_count_and_keep_historical_reference():
    reference = json.loads((launch.ROOT / "configs/research/round_scaling_j1_h1_200k.json").read_text())
    assert reference["shared_alg_params"]["inner_critic_updates_per_round"] == 32
    for count in launch.ACTOR_UPDATES:
        if count == 4:
            assert not (launch.ROOT / "configs/research/actor_steps_a4_j1_h1_200k.json").exists()
            continue
        actual = json.loads((launch.ROOT / f"configs/research/actor_steps_a{count}_j1_h1_200k.json").read_text())
        before, after = reference["shared_alg_params"], actual["shared_alg_params"]
        assert {k for k in set(before)|set(after) if before.get(k)!=after.get(k)} == {"inner_actor_updates_per_round"}
        assert after["inner_actor_updates_per_round"] == count
        assert after["inner_critic_updates_per_round"] == 32
        assert after["inner_actor_adaptation"] == after["inner_critic_adaptation"] == "clone"
        assert after["inner_actor_initialization"] == after["inner_critic_initialization"] == "prior"
        assert after["inner_temperature"] == 0
        assert after["inner_rounds"] == after["inner_rollout_horizon"] == 1
        assert actual["evaluation"] == reference["evaluation"]
        assert actual["checkpoint_contract"] == reference["checkpoint_contract"]
        assert actual["evaluation"]["togo_return_rollouts"] == 32


def test_tasks_cover_new_cells_once_and_never_rerun_a4(campaign):
    data = launch.load_campaign(campaign)
    tasks = launch.task_cells(data)
    assert len(tasks) == 36
    assert tasks == [(i,s) for i in (9,10,11,3,4,5,0,1,2) for s in range(4)]
    seen = set()
    for index, shard in tasks:
        cell = data["cells"][index]
        assert cell["actor_updates"] != 4 and not cell["reused"]
        for seed in cell["seed_shards"][shard]:
            key = cell["cell_id"], seed
            assert key not in seen
            seen.add(key)
    assert len(seen) == 9 * 20


@pytest.mark.parametrize("mutation",["counts","a4_rerun","new_reuse","rounds","seed","shard","output","spec","pin"])
def test_reject_protocol_drift_before_work(campaign, mutation):
    data=json.loads(campaign.read_text())
    if mutation=="counts": data["actor_updates"][-1]=16
    elif mutation=="a4_rerun": data["cells"][6]["reused"]=False
    elif mutation=="new_reuse": data["cells"][0]["reused"]=True
    elif mutation=="rounds": data["cells"][0]["rounds"]=2
    elif mutation=="seed": data["cells"][0]["seeds"]=[101]
    elif mutation=="shard": data["cells"][9]["seed_shards"][0].pop()
    elif mutation=="output": data["cells"][0]["bundle"]=data["cells"][1]["bundle"]
    elif mutation=="spec": data["cells"][0]["identity_spec"]=data["cells"][1]["identity_spec"]
    else: data["cells"][6]["bundle_seal_sha256"]="invalid"
    campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError): launch.load_campaign(campaign)
    assert not (campaign.parent/"production").exists()


@pytest.mark.parametrize("key,value",[("inner_actor_lr",.001),("inner_critic_updates_per_round",64),
    ("inner_temperature",.0001),("inner_critic_adaptation","frozen"),("inner_rollout_horizon",2)])
def test_reject_changes_beyond_actor_count(campaign,tmp_path,key,value):
    data=json.loads(campaign.read_text()); cell=data["cells"][0]
    matrix=json.loads((launch.ROOT/cell["matrix"]).read_text())
    matrix["shared_alg_params"][key]=value
    path=tmp_path/"invalid-matrix.json";path.write_text(json.dumps(matrix))
    cell["matrix"]=str(path);campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="Only actor update count"):
        launch.load_campaign(campaign)


@pytest.mark.parametrize("task,count",[(0,8),(12,2),(24,1)])
def test_optional_smokes_keep_c32_mean_protocol_and_never_rerun_a4(campaign,monkeypatch,task,count):
    calls=mock_runtime(monkeypatch,campaign.parent)
    launch.run_worker(campaign,task,smoke=True)
    command=next(c for c in calls if c[0]=="evaluate_ambi_checkpoint.py")
    assert f"actor_steps_a{count}_j1_h1_200k.json" in command[command.index("--matrix")+1]
    assert command[command.index("--controller-seed")+1]=="55"
    assert command[command.index("--seeds")+1]=="101"
    assert command[command.index("--max-steps")+1]=="2"
    assert "initialization/prior" in command and "initialization/inherited" in command
    assert "--reference-bundle" not in command  # Strict shard merge does not support external references.
    assert calls[-1][1]=="seal-episodes"
    assert not any(c[0] in ("eval_series.py","report_ambi_benchmark.py") for c in calls)
    data=launch.load_campaign(campaign);index,shard=launch.task_cells(data)[task]
    receipt=json.loads((launch._shard_path(data,data["cells"][index],shard,True)/"worker-completion.json").read_text())
    assert receipt["actor_updates"]==count and receipt["critic_updates"]==32 and receipt["rounds"]==1
    assert receipt["campaign_sha256"]==launch._hash(campaign)


def test_metadata_is_full_12_cell_readonly_preflight(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent)
    commands=launch.metadata(campaign)
    assert len(commands)==12 and calls==[]
    for item,cell in zip(commands,launch.load_campaign(campaign)["cells"]):
        assert item["cell_id"]==cell["cell_id"] and item["identity_spec"]==cell["identity_spec"]
        command=item["command"]
        assert "--eval-series-spec-dir" in command
        assert command[command.index("--bundle-dir")+1].endswith('/unused-bundle')


def test_merge_requires_all_production_shards(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent)
    with pytest.raises(ValueError,match="every seed shard"):
        launch.merge_cell(campaign,9,shard_indices=[0])
    assert calls==[]


def test_smoke_merge_validates_two_shards_and_stays_offline(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent)
    launch.run_worker(campaign,0,smoke=True)
    launch.run_worker(campaign,1,smoke=True)
    validated=[]
    monkeypatch.setattr(launch,"_validate_identity",lambda *args:validated.append(args))
    calls.clear();launch.merge_cell(campaign,9,smoke=True,shard_indices=[0,1])
    assert validated[0][-2:]==([101,106],True)
    assert calls[0][:2]==["merge_ambi_seed_shards.py","episodes"]
    assert calls[1][:2]==["evaluate_ambi_calibration.py","export-model"]
    assert not any(c[0]=="eval_series.py" for c in calls)
    receipt=json.loads((campaign.parent/"smoke/a8-s55/merge-completion.json").read_text())
    assert receipt["staged"] is False and receipt["actor_updates"]==8 and receipt["critic_updates"]==32 and receipt["rounds"]==1


def test_merge_rejects_receipt_after_manifest_changes(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent);launch.run_worker(campaign,24,smoke=True);calls.clear()
    data=json.loads(campaign.read_text());data["attempt_label"]="changed";campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="receipt differs"):
        launch.merge_cell(campaign,0,smoke=True,shard_indices=[0])
    assert calls==[]


def test_historical_a4_reuse_is_pinned_complete_immutable_and_never_executes(campaign,monkeypatch):
    from utils import ambi_seed_shards
    calls=mock_runtime(monkeypatch,campaign.parent)
    with pytest.raises(ValueError,match="cannot become a smoke"):
        launch.merge_cell(campaign,6,smoke=True)
    data=json.loads(campaign.read_text());cell=data["cells"][6]
    bundle=Path(cell["bundle"]);bundle.mkdir(parents=True);(bundle/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match="pinned manifest/seal"):
        launch.merge_cell(campaign,6)
    (bundle/ambi_seed_shards.SEAL).write_text('{}')
    cell['bundle_manifest_sha256']=launch._hash(bundle/'manifest.json')
    cell['bundle_seal_sha256']=launch._hash(bundle/ambi_seed_shards.SEAL);campaign.write_text(json.dumps(data))
    monkeypatch.setattr(ambi_seed_shards,'_read_episode',lambda path: ({},launch.SEEDS,{}))
    validated=[];monkeypatch.setattr(launch,'_validate_identity',lambda *args:validated.append(args))
    import sys
    from types import SimpleNamespace
    ambi_actor_steps_report = SimpleNamespace(load_cell=None)
    monkeypatch.setitem(sys.modules, "slurm.ambi_actor_steps_report", ambi_actor_steps_report)
    parsed=[]
    def parse_historical(cell):
        parsed.append(cell['cell_id'])
        return {'rows':[{'critic_training_steps':list(range(32)), 'actor_training_steps':list(range(4))} for _ in launch.SEEDS],
                'identity':{'training_trace_sources':list(range(20))}}
    monkeypatch.setattr(ambi_actor_steps_report,'load_cell',parse_historical)
    before={p.name:p.read_bytes() for p in bundle.iterdir()}
    def invalid_trace(cell):
        raise ValueError('Historical critic trace incomplete')
    with monkeypatch.context() as failing:
        failing.setattr(ambi_actor_steps_report,'load_cell',invalid_trace)
        with pytest.raises(ValueError,match='Historical critic trace'):
            launch.merge_cell(campaign,6)
    assert not (campaign.parent/'production/a4-s55/merge-completion.json').exists()
    launch.merge_cell(campaign,6)
    assert calls==[] and len(validated)==2
    assert {p.name:p.read_bytes() for p in bundle.iterdir()}==before
    receipt=json.loads((campaign.parent/'production/a4-s55/merge-completion.json').read_text())
    assert receipt['reused'] is True and receipt['staged'] is False
    assert (receipt['actor_updates'],receipt['critic_updates'],receipt['rounds']) == (4,32,1)
    assert parsed==['a4-s55']
    assert receipt['historical_validation']=={'episode_count':20,'critic_training_points':640,'actor_training_points':80,'trace_source_count':20}


def test_normalized_identity_must_match_current_science_and_protocol(campaign,monkeypatch):
    from utils import eval_series_data
    mock_runtime(monkeypatch,campaign.parent);data=launch.load_campaign(campaign);cell=data['cells'][6]
    identity=dict(protocol=dict(environment_seeds=launch.SEEDS,max_steps=500,controller_seed=55),
                  planner={'critic_updates':32},science={'source_sha256':'fixed'})
    spec=Path(cell['identity_spec']);spec.parent.mkdir(parents=True);spec.write_text(json.dumps({'identity':identity}))
    record=dict(selector=launch.SELECTOR,identity=copy.deepcopy(identity),checkpoint={'sha256':'c'*64})
    monkeypatch.setattr(eval_series_data,'normalize_bundle',lambda *a,**kw:[record])
    launch._validate_identity(data,cell,'bundle',launch.SEEDS,False)
    record['identity']['science']['source_sha256']='changed'
    with pytest.raises(ValueError,match='science/protocol'):
        launch._validate_identity(data,cell,'bundle',launch.SEEDS,False)


def test_launchers_lock_source_and_keep_workers_offline():
    assert 'eval_series.py' not in inspect.getsource(launch.run_worker)
    assert 'eval_series.py' not in inspect.getsource(launch.merge_cell)
    for suffix in ('oscar','merge_oscar','report_oscar'):
        text=(launch.ROOT/f'slurm/run_ambi_actor_steps_{suffix}.sbatch').read_text()
        assert 'EXPECTED_ACTION_MODES_SHA' in text and 'git status --porcelain' in text
        assert 'AMBI_ACTOR_STEPS_CAMPAIGN' in text
        assert 'environments/dmcontrol/.venv/bin/python' in text
        assert ('--mem=32G' if suffix=='oscar' else '--mem=16G') in text
        if suffix!='report_oscar': assert 'WANDB_MODE=offline' in text



def test_actor_dose_preserves_collection_critic_fit_and_actor_update_prefix(monkeypatch):
    """At the same root, A alone changes how long the paired actor phase runs."""
    import torch
    from RL.tdmpc2_core.inner_trace import InnerActionTrace
    from tests.test_ambi_root_local_sac import _tiny_component_model

    captures = {}
    for count in launch.ACTOR_UPDATES:
        model = _tiny_component_model(inner_rounds=1, inner_rollouts_per_round=8,
            inner_rollout_horizon=1, inner_batch_size=16, inner_replay_capacity=32,
            inner_critic_updates_per_round=32, inner_actor_updates_per_round=count,
            inner_actor_initialization='prior', inner_critic_initialization='prior',
            inner_critic_target_initialization='online', inner_temperature_mode='fixed',
            inner_temperature_initialization='fixed', inner_temperature=0.,
            inner_finite_horizon=True, inner_sac_critic_target='reward_only',
            inner_execution_action='mean', dropout=.01)
        engine = model.agent.inner_engine
        steps = []
        original_step = engine._sac_policy_step
        def observed_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            steps.append({k:v.detach().clone() for k,v in engine.state.actor.state_dict().items()})
            return result
        monkeypatch.setattr(engine, '_sac_policy_step', observed_step)
        trace = InnerActionTrace(probes=True, probe_mode='outer_tail', probe_rollouts=4,
                                 probe_horizon=1, probe_seed=991, capture_actors=True)
        try:
            with torch.no_grad():
                for head in model.agent.model._Qs.modules_list:
                    head[-1].weight.normal_(0., .1)
            outer = {k:v.clone() for k,v in model.agent.model.state_dict().items()}
            rng = torch.random.get_rng_state().clone()
            model.agent.act(torch.zeros(3), t0=True, eval_mode=True, trace=trace)
            pool = engine._action_pool
            assert model.agent.last_inner_metrics['inner_actor_optimizer_steps'] == count
            assert model.agent.last_inner_metrics['inner_critic_optimizer_steps'] == 32
            assert model.agent.last_inner_metrics['inner_critic_target_updates'] == 32
            assert model.agent.last_inner_metrics['inner_alpha'] == 0
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            for k,v in model.agent.model.state_dict().items():
                torch.testing.assert_close(v, outer[k], rtol=0, atol=0)
            initial = trace.actor_snapshots[0].make_policy().state_dict()
            for k,v in model.agent.model._pi.state_dict().items():
                torch.testing.assert_close(initial[k], v, rtol=0, atol=0)
            assert len(steps) == count
            assert any(not torch.equal(v, initial[k]) for k,v in steps[-1].items())
            probes = [e for e in trace.events if e['phase'] == 'probe']
            assert [(e['round_index'], e['actor_updates'], e['critic_updates']) for e in probes] == [(0,0,0),(1,count,32)]
            updates = [e for e in trace.events if e['phase'] == 'update']
            assert [e['updated_critic'] for e in updates] == [True]*32 + [False]*count
            assert [e['updated_actor'] for e in updates] == [False]*32 + [True]*count
            replay = pool.replay
            assert replay.size == 8 and replay.horizon_end[:8].eq(1).all()
            torch.testing.assert_close(replay.z[:8], replay.z[:1].expand(8,-1), rtol=0, atol=0)
            captures[count] = dict(steps=steps, critic={k:v.clone() for k,v in pool.critic.state_dict().items()},
                replay={k:getattr(replay,k)[:replay.size].clone() for k in ('z','action','reward','next_z','terminated','horizon_end')})
        finally:
            model.env.close()
    for count, captured in captures.items():
        for name in ('critic', 'replay'):
            for k,v in captured[name].items():
                torch.testing.assert_close(v, captures[8][name][k], rtol=0, atol=0)
        for index, state in enumerate(captured['steps']):
            for k,v in state.items():
                torch.testing.assert_close(v, captures[8]['steps'][index][k], rtol=0, atol=0)
