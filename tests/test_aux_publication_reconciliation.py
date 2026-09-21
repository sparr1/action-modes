import pytest
from slurm.reconcile_aux_training_publication import missing_history


def test_only_missing_suffix_is_replayed():
    expected = [{'axis/decision':i, 'metric':i*.1} for i in range(4)]
    actual = [{'_step':i, **expected[i], '_timestamp':123} for i in range(2)]
    assert missing_history(expected, actual) == expected[2:]
    assert missing_history(expected, [{'_step':i, **r} for i,r in enumerate(expected)]) == []


@pytest.mark.parametrize('actual', [
    [{'_step':1,'metric':2}],
    [{'_step':0,'metric':1},{'_step':0,'metric':1}],
    [{'_step':0,'metric':99}],
    [{'_step':0}],
])
def test_uncertain_or_conflicting_history_is_not_overwritten(actual):
    with pytest.raises(AssertionError): missing_history([{'metric':1},{'metric':2}],actual)


def test_recovery_reads_full_diagnostic_bundle_not_manifest_only(tmp_path):
    from tests.test_ambi_diagnostic_series import record
    from utils.ambi_diagnostic_series import write_diagnostic_bundle, diagnostic_history
    from slurm.reconcile_aux_training_publication import saved_diagnostic_history
    expected = record(bootstrap_resamples=10)
    write_diagnostic_bundle(tmp_path/'model-series', expected)
    loaded, rows = saved_diagnostic_history(tmp_path/'model-series')
    assert loaded == expected and rows == diagnostic_history(expected)
