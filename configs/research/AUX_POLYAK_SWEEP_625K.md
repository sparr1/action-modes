# Faster inner target updates at625k

`ambi_aux_polyak_sweep_625k.json` repeats the two soft-initialized critic arms
of the H/J sweep with `inner_critic_target_tau=0.1`, compared with the existing
0.01 results. Polyak updates use target=(1-tau)*target+tau*online, so this
increases target tracking speed. No algorithm implementation changes are made.

The18conditions are soft/soft and soft/return, H={1,2,3}, J={1,2,4}. Each runs
five complete500-decision episodes, seeds101–105/controller55, on the exact
625k auxiliary-return seed55 checkpoint. N128/B256/C32/A4, all replay retained,
automatic inherited actor temperature, target initialization and update cadence,
critic dropout, optimizer settings, execution and diagnostics match the earlier
sweep. Soft fitting remains entropy-augmented; the soft outer boundary includes
frozen outer entropy and the return boundary does not. H1 is retained as a
negative control because its targets always select the frozen outer boundary.

Use the existing H/J runner with explicit matrix, group and baseline campaign:

```bash
python slurm/ambi_aux_hj_sweep.py prepare --root "$ROOT" \
  --matrix configs/research/ambi_aux_polyak_sweep_625k.json \
  --group aux625k-polyak010-h123-j124-20260917 \
  --label '625k Polyak tau0.1' --baseline-campaign "$PREVIOUS_HJ_ROOT" \
  --checkpoint "$CHECKPOINT" --inventory "$INVENTORY" \
  --reference "$PRIOR_REFERENCE" --registry "$REGISTRY"
```

Preparation validates that each matched baseline has the same backbone,
scientific implementation, protocol and resolved planner except target tau.
The frozen-prior and tau0.01 measurements are reused, not rerun. Each new setting
has a new performance identity and training diagnostic run. The ordinary
performance curve retains gain versus prior. Training runs also log
`comparison/tau001_gain_mean`, paired sample SD and exploratory95% paired
bootstrap intervals (2000 resamples, seed20260912), with five per-seed deltas in
`polyak-comparison.json`. The overview shows this direct tau comparison.

GPU workers and the CPU watcher read the campaign's saved matrix/group. The
existing Slurm worker/watch launchers are reused with18array cells, submitting
J4 first and using live account capacity. Two H3/J4 smoke cases cover both
boundary objectives before release. One GPU owns each complete five-seed panel.

All update metrics and raw traces remain available as in AUX_HJ_SWEEP_625K.md.
W&B performance acknowledgement permits120seconds and bounded retries through
the existing identity/hash/SDK-slot reconciliation. It never blindly resends an
uncertain row or resumes uncertain training-diagnostic publication.
