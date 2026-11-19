"""Reject stale or altered Q10 validation evidence before releasing training."""
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest
from slurm.ambi_transfer_training import ROOT, binding
from slurm.release_dog_walk_mrsq import validate_gate_files


@pytest.fixture
def evidence(tmp_path):
    manifest = ROOT / 'configs/dmcontrol/experiments/ambi_dog_walk_mrsq_q10_seed55.json'
    receipt = binding(manifest)
    parity = tmp_path / 'parity.xml'
    parity.write_text('<testsuite tests="2"/>')
    gate = {'binding': receipt, 'cuda_parity_junit': str(parity),
            'cuda_parity_sha256': hashlib.sha256(parity.read_bytes()).hexdigest()}
    return gate, {'commit': receipt['commit']}, tmp_path


def test_unchanged_pinned_files_accept_evidence(evidence):
    gate, state, root = evidence
    validate_gate_files(gate, state, ROOT, root)


@pytest.mark.parametrize('mismatch', ('config', 'lock', 'manifest', 'commit', 'parity'))
def test_changed_evidence_rejected(evidence, mismatch):
    original, state, root = evidence
    gate = deepcopy(original)
    if mismatch == 'config':
        gate['binding']['config_sha256']['ambi_dog_walk_mrsq_q10_seed55'] = '0' * 64
    elif mismatch == 'lock':
        gate['binding']['lock_sha256'] = '0' * 64
    elif mismatch == 'manifest':
        gate['binding']['manifest'] = 'configs/dmcontrol/experiments/ambi_dog_walk_mrsq_q5_seed55.json'
    elif mismatch == 'commit':
        gate['binding']['commit'] = '0' * 40
    else:
        Path(gate['cuda_parity_junit']).write_text('<testsuite tests="0"/>')
    with pytest.raises(AssertionError):
        validate_gate_files(gate, state, ROOT, root)
