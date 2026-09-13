"""C64 round extension, immutable references, and round-persistent optimizers."""
import copy
import inspect
import json
from pathlib import Path

import pytest

from slurm import ambi_c64_rounds_campaign as launch


@pytest.fixture
def campaign(tmp_path):
    cells = []
    for count in launch.CRITIC_UPDATES:
        for rounds in launch.ROUNDS:
            for seed in launch.CONTROLLERS:
                cell_id = f"c{count}-j{rounds}-s{seed}"
                reused = count == 32 or rounds == 1
                output = tmp_path / ("historical" if reused else "production") / cell_id
                matrix = (f"round_scaling_j{rounds}_h1_200k.json" if count == 32 else
                          "critic_steps_c64_j1_h1_200k.json" if rounds == 1 else f"c64_rounds_j{rounds}_h1_200k.json")
                width = 5 if rounds > 1 else 10
                cells.append(dict(cell_id=cell_id, critic_updates=count, rounds=rounds, controller_seed=seed,
                    matrix=f"configs/research/{matrix}", selector=launch.SELECTOR,
                    seeds=launch.SEEDS, seed_shards=[launch.SEEDS[i:i+width] for i in range(0,20,width)],
                    reused=reused, bundle=str(output / "bundle"), model_series=str(output / "model-series"),
                    identity_spec=str(tmp_path / "specs" / cell_id / "initialization__inherited.json"),
                    **({"bundle_manifest_sha256":"a"*64,"bundle_seal_sha256":"b"*64} if reused else {})))
    data = dict(schema_version=1, attempt_label="test-c64-rounds", output_root=str(tmp_path),
                inventory=str(tmp_path / "inventory.json"), critic_updates=launch.CRITIC_UPDATES, rounds=launch.ROUNDS,
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


def test_new_matrices_change_only_rounds_from_completed_c64_j1():
    reference = json.loads((launch.ROOT / "configs/research/critic_steps_c64_j1_h1_200k.json").read_text())
    for rounds in (2,):
        actual = json.loads((launch.ROOT / f"configs/research/c64_rounds_j{rounds}_h1_200k.json").read_text())
        before, after = reference["shared_alg_params"], actual["shared_alg_params"]
        assert {k for k in set(before)|set(after) if before.get(k)!=after.get(k)} == {"inner_rounds"}
        assert after["inner_rounds"] == rounds
        assert after["inner_critic_updates_per_round"] == 64
        assert after["inner_actor_updates_per_round"] == 4
        assert after["inner_actor_adaptation"] == after["inner_critic_adaptation"] == "clone"
        assert after["inner_actor_initialization"] == after["inner_critic_initialization"] == "prior"
        assert after["inner_temperature"] == 0 and after["inner_rollout_horizon"] == 1
        assert actual["evaluation"] == reference["evaluation"]
        assert actual["checkpoint_contract"] == reference["checkpoint_contract"]
        assert actual["oscar_seed_shards"] == [list(range(i,i+5)) for i in range(101,121,5)]


def test_single_twelve_task_wave_covers_only_new_c64_j2(campaign):
    data = launch.load_campaign(campaign)
    tasks = launch.task_cells(data)
    assert len(tasks) == 12
    assert tasks == [(i,s) for i in (9,10,11) for s in range(4)]
    seen = set()
    for index, shard in tasks:
        cell = data["cells"][index]
        assert cell["critic_updates"] == 64 and cell["rounds"] == 2 and not cell["reused"]
        assert len(cell["seed_shards"][shard]) == 5
        for seed in cell["seed_shards"][shard]:
            key = cell["cell_id"], seed
            assert key not in seen
            seen.add(key)
    assert len(seen) == 3 * 20


@pytest.mark.parametrize("mutation",["counts","c32_rerun","c64_j1_rerun","new_reuse","rounds","seed","shard","output","spec","pin"])
def test_reject_protocol_drift_before_work(campaign, mutation):
    data=json.loads(campaign.read_text())
    if mutation=="counts": data["critic_updates"][-1]=128
    elif mutation=="c32_rerun": data["cells"][0]["reused"]=False
    elif mutation=="c64_j1_rerun": data["cells"][6]["reused"]=False
    elif mutation=="new_reuse": data["cells"][9]["reused"]=True
    elif mutation=="rounds": data["cells"][0]["rounds"]=8
    elif mutation=="seed": data["cells"][0]["seeds"]=[101]
    elif mutation=="shard": data["cells"][9]["seed_shards"][0].pop()
    elif mutation=="output": data["cells"][0]["bundle"]=data["cells"][1]["bundle"]
    elif mutation=="spec": data["cells"][0]["identity_spec"]=data["cells"][1]["identity_spec"]
    else: data["cells"][0]["bundle_seal_sha256"]="invalid"
    campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError): launch.load_campaign(campaign)
    assert not (campaign.parent/"production").exists()


@pytest.mark.parametrize("key,value",[("inner_actor_lr",.001),("inner_actor_updates_per_round",8),
    ("inner_temperature",.0001),("inner_critic_adaptation","frozen"),("inner_rollout_horizon",2)])
def test_reject_changes_beyond_critic_count_and_rounds(campaign,tmp_path,key,value):
    data=json.loads(campaign.read_text()); cell=data["cells"][0]
    matrix=json.loads((launch.ROOT/cell["matrix"]).read_text())
    matrix["shared_alg_params"][key]=value
    path=tmp_path/"invalid-matrix.json";path.write_text(json.dumps(matrix))
    cell["matrix"]=str(path);campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="Only critic update count and rounds"):
        launch.load_campaign(campaign)


@pytest.mark.parametrize("task,rounds,seed",[(0,2,101),(1,2,106)])
def test_smokes_cover_only_j2_and_second_seed_shard(campaign,monkeypatch,task,rounds,seed):
    calls=mock_runtime(monkeypatch,campaign.parent)
    launch.run_worker(campaign,task,smoke=True)
    command=next(c for c in calls if c[0]=="evaluate_ambi_checkpoint.py")
    assert f"c64_rounds_j{rounds}_h1_200k.json" in command[command.index("--matrix")+1]
    assert command[command.index("--controller-seed")+1]=="55"
    assert command[command.index("--seeds")+1]==str(seed)
    assert command[command.index("--max-steps")+1]=="2"
    assert "initialization/prior" in command and "initialization/inherited" in command
    assert "--reference-bundle" not in command
    assert calls[-1][1]=="seal-episodes"
    data=launch.load_campaign(campaign);index,shard=launch.task_cells(data)[task]
    receipt=json.loads((launch._shard_path(data,data["cells"][index],shard,True)/"worker-completion.json").read_text())
    assert receipt["critic_updates"]==64 and receipt["rounds"]==rounds
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
    receipt=json.loads((campaign.parent/"smoke/c64-j2-s55/merge-completion.json").read_text())
    assert receipt["staged"] is False and receipt["critic_updates"]==64


def test_merge_rejects_receipt_after_manifest_changes(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent);launch.run_worker(campaign,0,smoke=True);calls.clear()
    data=json.loads(campaign.read_text());data["attempt_label"]="changed";campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="receipt differs"):
        launch.merge_cell(campaign,9,smoke=True,shard_indices=[0])
    assert calls==[]


@pytest.mark.parametrize("cell_index",[0,3,6])
def test_historical_reuse_is_pinned_complete_immutable_and_never_executes(campaign,monkeypatch,cell_index):
    from utils import ambi_seed_shards
    calls=mock_runtime(monkeypatch,campaign.parent)
    with pytest.raises(ValueError,match="cannot become a smoke"):
        launch.merge_cell(campaign,cell_index,smoke=True)
    data=json.loads(campaign.read_text());cell=data["cells"][cell_index]
    bundle=Path(cell["bundle"]);bundle.mkdir(parents=True);(bundle/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match="pinned manifest/seal"):
        launch.merge_cell(campaign,cell_index)
    (bundle/ambi_seed_shards.SEAL).write_text('{}')
    cell['bundle_manifest_sha256']=launch._hash(bundle/'manifest.json')
    cell['bundle_seal_sha256']=launch._hash(bundle/ambi_seed_shards.SEAL);campaign.write_text(json.dumps(data))
    monkeypatch.setattr(ambi_seed_shards,'_read_episode',lambda path: ({},launch.SEEDS,{}))
    validated=[];monkeypatch.setattr(launch,'_validate_identity',lambda *args:validated.append(args))
    from slurm import ambi_c64_rounds_report
    parsed=[]
    def parse_historical(cell):
        parsed.append(cell['cell_id'])
        return {'rows':[{'critic_training_steps':list(range(cell['critic_updates']*cell['rounds']))} for _ in launch.SEEDS],
                'identity':{'training_trace_sources':list(range(20))}}
    monkeypatch.setattr(ambi_c64_rounds_report,'load_cell',parse_historical)
    before={p.name:p.read_bytes() for p in bundle.iterdir()}
    def invalid_trace(cell):
        raise ValueError('Historical critic trace incomplete')
    with monkeypatch.context() as failing:
        failing.setattr(ambi_c64_rounds_report,'load_cell',invalid_trace)
        with pytest.raises(ValueError,match='Historical critic trace'):
            launch.merge_cell(campaign,cell_index)
    assert not (campaign.parent/'production'/cell['cell_id']/'merge-completion.json').exists()
    launch.merge_cell(campaign,cell_index)
    assert calls==[] and len(validated)==2
    assert {p.name:p.read_bytes() for p in bundle.iterdir()}==before
    receipt=json.loads((campaign.parent/'production'/cell['cell_id']/'merge-completion.json').read_text())
    assert receipt['reused'] is True and receipt['staged'] is False
    assert receipt['rounds']==cell['rounds'] and receipt['critic_updates']==cell['critic_updates']
    assert parsed==[cell['cell_id']]
    assert receipt['historical_validation']=={'episode_count':20,'critic_training_points':20*cell['critic_updates']*cell['rounds'],'trace_source_count':20}


def test_normalized_identity_must_match_current_science_and_protocol(campaign,monkeypatch):
    from utils import eval_series_data
    mock_runtime(monkeypatch,campaign.parent);data=launch.load_campaign(campaign);cell=data['cells'][0]
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
        text=(launch.ROOT/f'slurm/run_ambi_c64_rounds_{suffix}.sbatch').read_text()
        assert 'EXPECTED_ACTION_MODES_SHA' in text and 'git status --porcelain' in text
        assert 'AMBI_C64_ROUNDS_CAMPAIGN' in text
        assert 'environments/dmcontrol/.venv/bin/python' in text
        assert ('--mem=32G' if suffix=='oscar' else '--mem=16G') in text
        if suffix!='report_oscar': assert 'WANDB_MODE=offline' in text




def test_c64_rounds_preserve_optimizers_replay_and_common_prefix_then_reset(monkeypatch):
    import torch
    from RL.tdmpc2_core.inner_trace import InnerActionTrace
    from tests.test_ambi_root_local_sac import _tiny_component_model
    prefixes=[]
    for rounds in (1,2):
        model=_tiny_component_model(inner_rounds=rounds,inner_rollouts_per_round=8,inner_rollout_horizon=1,
            inner_batch_size=16,inner_replay_capacity=64,inner_critic_updates_per_round=64,
            inner_actor_updates_per_round=4,inner_actor_initialization='prior',inner_critic_initialization='prior',
            inner_critic_target_initialization='online',inner_temperature_mode='fixed',
            inner_temperature_initialization='fixed',inner_temperature=0.,inner_finite_horizon=True,
            inner_sac_critic_target='reward_only',inner_execution_action='mean',dropout=.01)
        engine=model.agent.inner_engine;entries=[];original=engine._run_component_update_counts
        def observe(**kwargs):
            state=engine.state
            def steps(optimizer):
                return {int(v['step']) for v in optimizer.state.values() if 'step' in v} or {0}
            entries.append({'critic':state.critic_steps,'actor':state.actor_steps,
                'actor_optimizer':state.actor_optim,'critic_optimizer':state.critic_optim,
                'actor_adam_steps':steps(state.actor_optim),'critic_adam_steps':steps(state.critic_optim),
                'replay_size':state.replay.size})
            return original(**kwargs)
        monkeypatch.setattr(engine,'_run_component_update_counts',observe)
        trace=InnerActionTrace(probes=True,probe_mode='outer_tail',probe_rollouts=4,probe_horizon=1,
                              probe_seed=882,capture_actors=True)
        try:
            outer={k:v.clone() for k,v in model.agent.model.state_dict().items()}
            rng=torch.random.get_rng_state().clone()
            model.agent.act(torch.zeros(3),t0=True,eval_mode=True,trace=trace)
            assert len(entries)==rounds
            for r,x in enumerate(entries):
                assert x['critic']==64*r and x['actor']==4*r and x['replay_size']==8*(r+1)
                assert x['actor_adam_steps']=={4*r} and x['critic_adam_steps']=={64*r}
                assert x['actor_optimizer'] is entries[0]['actor_optimizer']
                assert x['critic_optimizer'] is entries[0]['critic_optimizer']
            probes=[e for e in trace.events if e['phase']=='probe']
            assert [(e['round_index'],e['actor_updates'],e['critic_updates']) for e in probes]==[(r,4*r,64*r) for r in range(rounds+1)]
            for r in range(1,rounds+1):
                updates=[e for e in trace.events if e['phase']=='update' and e['round_index']==r]
                assert [e['updated_critic'] for e in updates]==[True]*64+[False]*4
                assert [e['updated_actor'] for e in updates]==[False]*64+[True]*4
            prefixes.append({'actors':[x.make_policy().state_dict() for x in trace.actor_snapshots[:2]],
                             'probes':[{k:v for k,v in e['metrics'].items() if k.startswith('togo_')} for e in probes[:2]]})
            torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
            # A new real decision reuses allocations but clears every scientific state.
            next_trace=InnerActionTrace(capture_actors=True)
            model.agent.act(torch.zeros(3),t0=False,eval_mode=True,trace=next_trace)
            assert len(entries)==2*rounds
            first=entries[rounds]
            assert first['actor']==first['critic']==0 and first['replay_size']==8
            assert first['actor_adam_steps']==first['critic_adam_steps']=={0}
            for k,v in next_trace.actor_snapshots[0].make_policy().state_dict().items():
                torch.testing.assert_close(v,model.agent.model._pi.state_dict()[k],rtol=0,atol=0)
            for k,v in model.agent.model.state_dict().items():torch.testing.assert_close(v,outer[k],rtol=0,atol=0)
            torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
        finally:model.env.close()
    # Increasing the stopping round alone preserves the first round at one root.
    for a,b in zip(prefixes[0]['actors'],prefixes[1]['actors']):
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    assert prefixes[0]['probes']==prefixes[1]['probes']
