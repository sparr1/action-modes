"""Only entropy varies in the existing full-episode J1/J2/J4 protocol."""
import copy
import inspect
import json
from pathlib import Path

import pytest

from slurm import ambi_entropy_episode_campaign as launch


@pytest.fixture
def campaign(tmp_path):
    cells = []
    for arm in launch.ARMS:
        for j in (1, 2, 4):
            for seed in launch.CONTROLLERS:
                cell_id = f"{arm}-j{j}-c{seed}"
                output = tmp_path / ("historical" if arm == "off" else "production") / cell_id
                matrix = f"round_scaling_j{j}_h1_200k.json" if arm == "off" else f"entropy_{arm}_j{j}_h1_200k.json"
                cells.append(dict(cell_id=cell_id, arm=arm, rounds=j, controller_seed=seed,
                    matrix=f"configs/research/{matrix}", selector=launch.SELECTOR,
                    seeds=launch.SEEDS, seed_shards=[launch.SEEDS[i:i+20//j] for i in range(0,20,20//j)],
                    reused=arm == "off", bundle=str(output / "bundle"), model_series=str(output / "model-series"),
                    identity_spec=str(tmp_path / "specs" / cell_id / "initialization__inherited.json"),
                    **({"bundle_manifest_sha256":"a"*64,"bundle_seal_sha256":"b"*64} if arm == "off" else {})))
    data = dict(schema_version=1, attempt_label="test-entropy-episodes", output_root=str(tmp_path),
                inventory=str(tmp_path / "inventory.json"), arms=launch.ARMS, seeds=launch.SEEDS,
                controller_seeds=launch.CONTROLLERS, cells=cells)
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


def test_exact_matrices_change_only_entropy_and_keep_off_bitwise_config():
    for j in (1,2,4):
        reference = json.loads((launch.ROOT / f"configs/research/round_scaling_j{j}_h1_200k.json").read_text())
        for arm in ("native","squashed"):
            actual = json.loads((launch.ROOT / f"configs/research/entropy_{arm}_j{j}_h1_200k.json").read_text())
            before, after = reference["shared_alg_params"], actual["shared_alg_params"]
            changes = {key for key in set(before)|set(after) if before.get(key)!=after.get(key)}
            assert changes == ({"inner_temperature"} if arm=="native" else {"inner_temperature","inner_actor_entropy_mode"})
            assert after["inner_temperature"] == launch.ARMS[arm]["alpha"]
            assert after["inner_actor_entropy_mode"] == launch.ARMS[arm]["mode"]
            assert actual["evaluation"] == reference["evaluation"]
            assert actual["checkpoint_contract"] == reference["checkpoint_contract"]
            assert after["inner_critic_updates_per_round"] == 32
            assert after["inner_actor_updates_per_round"] == 4
            assert actual["evaluation"]["togo_return_rollouts"] == 32


def test_tasks_cover_all_new_cells_once_and_never_rerun_off(campaign):
    data = launch.load_campaign(campaign)
    tasks = launch.task_cells(data)
    assert len(tasks)==42
    assert tasks[:4]==[(15,i) for i in range(4)]
    assert tasks[12:16]==[(24,i) for i in range(4)]
    assert tasks[24:26]==[(12,0),(12,1)]
    assert tasks[30:32]==[(21,0),(21,1)]
    assert tasks[36:]==[(i,0) for i in (9,10,11,18,19,20)]
    seen=set()
    for index,shard in tasks:
        cell=data["cells"][index]
        assert cell["arm"]!="off" and not cell["reused"]
        assert cell["rounds"]*len(cell["seed_shards"][shard])==20
        for seed in cell["seed_shards"][shard]:
            key=cell["cell_id"],seed
            assert key not in seen
            seen.add(key)
    assert len(seen)==18*20


@pytest.mark.parametrize("mutation",["alpha","off_rerun","new_reuse","seed","shard","output","spec","pin"])
def test_reject_protocol_drift_before_any_work(campaign,mutation):
    data=json.loads(campaign.read_text())
    if mutation=="alpha": data["arms"]["native"]["alpha"] = .1
    elif mutation=="off_rerun": data["cells"][0]["reused"]=False
    elif mutation=="new_reuse": data["cells"][9]["reused"]=True
    elif mutation=="seed": data["cells"][9]["seeds"]=[101]
    elif mutation=="shard": data["cells"][9]["seed_shards"][0].pop()
    elif mutation=="output": data["cells"][9]["bundle"]=data["cells"][10]["bundle"]
    elif mutation=="spec": data["cells"][9]["identity_spec"]=data["cells"][10]["identity_spec"]
    else: data["cells"][0]["bundle_seal_sha256"]="invalid"
    campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError): launch.load_campaign(campaign)
    assert not (campaign.parent/"production").exists()


def test_config_changes_beyond_entropy_rejected(campaign,tmp_path):
    data=json.loads(campaign.read_text())
    cell=data["cells"][9]
    matrix=json.loads((launch.ROOT/cell["matrix"]).read_text())
    matrix["shared_alg_params"]["inner_actor_lr"]*=2
    path=tmp_path/"invalid-matrix.json";path.write_text(json.dumps(matrix))
    cell["matrix"]=str(path);campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="Only entropy"):
        launch.load_campaign(campaign)


@pytest.mark.parametrize("task,arm,j",[(0,"native",4),(12,"squashed",4),(24,"native",2),(30,"squashed",2),(36,"native",1),(39,"squashed",1)])
def test_six_smokes_preserve_nonzero_entropy_and_mean_episode_protocol(campaign,monkeypatch,task,arm,j):
    calls=mock_runtime(monkeypatch,campaign.parent)
    launch.run_worker(campaign,task,smoke=True)
    command=next(c for c in calls if c[0]=="evaluate_ambi_checkpoint.py")
    assert f"entropy_{arm}_j{j}_h1_200k.json" in command[command.index("--matrix")+1]
    assert command[command.index("--controller-seed")+1]=="55"
    assert command[command.index("--seeds")+1]=="101"
    assert command[command.index("--max-steps")+1]=="2"
    assert "initialization/prior" in command and "initialization/inherited" in command
    assert calls[-1][1]=="seal-episodes"
    assert not any(c[0] in ("eval_series.py","report_ambi_benchmark.py") for c in calls)
    data=launch.load_campaign(campaign)
    index,shard=launch.task_cells(data)[task]
    receipt=json.loads((launch._shard_path(data,data["cells"][index],shard,True)/"worker-completion.json").read_text())
    assert receipt["arm"]==arm and receipt["rounds"]==j
    assert receipt["campaign_sha256"]==launch._hash(campaign)


def test_metadata_is_full_27_cell_readonly_preflight(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent)
    commands=launch.metadata(campaign)
    assert len(commands)==27 and calls==[]
    for item,cell in zip(commands,launch.load_campaign(campaign)["cells"]):
        assert item["cell_id"]==cell["cell_id"]
        assert item["identity_spec"]==cell["identity_spec"]
        command=item["command"]
        assert "--eval-series-spec-dir" in command
        assert command[command.index("--bundle-dir")+1].endswith('/unused-bundle')


def test_merge_requires_all_production_shards(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent)
    with pytest.raises(ValueError,match="every seed shard"):
        launch.merge_cell(campaign,12,shard_indices=[0])
    assert calls==[]


def test_smoke_merge_validates_selected_shards_and_stays_offline(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent)
    launch.run_worker(campaign,24,smoke=True)
    launch.run_worker(campaign,25,smoke=True)
    validated=[]
    monkeypatch.setattr(launch,"_validate_identity",lambda *args:validated.append(args))
    calls.clear()
    launch.merge_cell(campaign,12,smoke=True)
    assert validated[0][-2:]==([101,111],True)
    assert calls[0][:2]==["merge_ambi_seed_shards.py","episodes"]
    assert calls[1][:2]==["evaluate_ambi_calibration.py","export-model"]
    assert not any(c[0]=="eval_series.py" for c in calls)
    receipt=json.loads((campaign.parent/"smoke/native-j2-c55/merge-completion.json").read_text())
    assert receipt["staged"] is False and receipt["arm"]=="native"


def test_merge_rejects_receipt_after_manifest_changed(campaign,monkeypatch):
    calls=mock_runtime(monkeypatch,campaign.parent)
    launch.run_worker(campaign,36,smoke=True)
    calls.clear()
    data=json.loads(campaign.read_text());data["attempt_label"]="changed";campaign.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="receipt differs"):
        launch.merge_cell(campaign,9,smoke=True)
    assert calls==[]


def test_historical_reuse_requires_pinned_seal_complete_identity_and_never_executes(campaign,monkeypatch):
    from utils import ambi_seed_shards
    calls=mock_runtime(monkeypatch,campaign.parent)
    with pytest.raises(ValueError,match="cannot become a smoke"):
        launch.merge_cell(campaign,0,smoke=True)
    data=json.loads(campaign.read_text());cell=data["cells"][0]
    bundle=Path(cell["bundle"]);bundle.mkdir(parents=True)
    (bundle/'manifest.json').write_text('{}')
    with pytest.raises(ValueError,match="pinned manifest/seal"):
        launch.merge_cell(campaign,0)
    (bundle/ambi_seed_shards.SEAL).write_text('{}')
    cell['bundle_manifest_sha256']=launch._hash(bundle/'manifest.json')
    cell['bundle_seal_sha256']=launch._hash(bundle/ambi_seed_shards.SEAL)
    campaign.write_text(json.dumps(data))
    monkeypatch.setattr(ambi_seed_shards,'_read_episode',lambda path: ({},launch.SEEDS,{}))
    validated=[]
    monkeypatch.setattr(launch,'_validate_identity',lambda *args:validated.append(args))
    before={p.name:p.read_bytes() for p in bundle.iterdir()}
    launch.merge_cell(campaign,0)
    assert calls==[] and len(validated)==1
    assert {p.name:p.read_bytes() for p in bundle.iterdir()}==before
    receipt=json.loads((campaign.parent/'production/off-j1-c55/merge-completion.json').read_text())
    assert receipt['reused'] is True and receipt['staged'] is False


def test_normalized_reuse_identity_must_equal_full_current_spec(campaign,monkeypatch):
    from utils import eval_series_data
    mock_runtime(monkeypatch,campaign.parent)
    data=launch.load_campaign(campaign);cell=data['cells'][0]
    identity=dict(protocol=dict(environment_seeds=launch.SEEDS,max_steps=500,controller_seed=55),
                  planner={'entropy':'off'},science={'source_sha256':'fixed'})
    spec=Path(cell['identity_spec']);spec.parent.mkdir(parents=True);spec.write_text(json.dumps({'identity':identity}))
    record=dict(selector=launch.SELECTOR,identity=copy.deepcopy(identity),checkpoint={'sha256':'c'*64})
    monkeypatch.setattr(eval_series_data,'normalize_bundle',lambda *a,**kw:[record])
    launch._validate_identity(data,cell,'bundle',launch.SEEDS,False)
    record['identity']['science']['source_sha256']='changed'
    with pytest.raises(ValueError,match='science/protocol'):
        launch._validate_identity(data,cell,'bundle',launch.SEEDS,False)


def test_launchers_lock_source_runtime_and_never_publish_individual_runs():
    assert 'eval_series.py' not in inspect.getsource(launch.run_worker)
    assert 'eval_series.py' not in inspect.getsource(launch.merge_cell)
    for suffix in ('oscar','merge_oscar'):
        text=(launch.ROOT/f'slurm/run_ambi_entropy_episodes_{suffix}.sbatch').read_text()
        assert 'EXPECTED_ACTION_MODES_SHA' in text and 'git status --porcelain' in text
        assert 'AMBI_ENTROPY_EPISODES_CAMPAIGN' in text and 'WANDB_MODE=offline' in text
        assert 'environments/dmcontrol/.venv/bin/python' in text
