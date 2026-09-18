"""Bounded, unpublished checkpoint diagnostic for inner-loop reproducibility.

Reuses the evaluator's initialization and warmup. Repeats the same episode
prefix, toggling only the replay reset flag; records tensor fingerprints at
eager boundaries around production compiled regions. Never trains the outer
model or writes to W&B. Run GPU work only in an allocated compute job.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import shutil

import numpy as np
import torch

from evaluate_ambi_checkpoint import (
    _digest_update, _file_sha256, _initialize_frozen_model, _make_env,
    _outer_state_digest, _seed_spaces,
)
from RL.tdmpc2_core.inner_trace import InnerActionTrace
from utils.ambi_benchmark import solver_seed


def digest(value):
    h = hashlib.sha256()
    _digest_update(h, value)
    return h.hexdigest()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def fingerprints(value, prefix=''):
    if isinstance(value, dict):
        return {p: v for k, item in value.items()
                for p, v in fingerprints(item, prefix + '/' + str(k)).items()}
    if isinstance(value, (tuple, list)):
        return {p: v for k, item in enumerate(value)
                for p, v in fingerprints(item, prefix + '/' + str(k)).items()}
    return {prefix: digest(value)}


def compiler_kernels(output=None):
    """Record the actual selected Triton launch configurations, when present."""
    from torch._inductor.codecache import PyCodeCache
    result = []
    for key, module in PyCodeCache.cache.items():
        if output is not None and getattr(module, '__file__', None):
            output.mkdir(exist_ok=True)
            destination = output / (key + '.py')
            if not destination.exists():
                shutil.copyfile(module.__file__, destination)
        for name, value in vars(module).items():
            if type(value).__name__ not in ('CachingAutotuner', 'DebugAutotuner'):
                continue
            result.append(dict(module=key, name=name,
                source_sha256=hashlib.sha256(str(value.fn.src).encode()).hexdigest(),
                launchers=[dict(kwargs=x.config.kwargs, num_warps=x.config.num_warps,
                                num_stages=x.config.num_stages) for x in value.launchers]))
    return result


class Recorder:
    def __init__(self, output):
        self.output = output
        self.events = []
        self.counts = {}

    def record(self, name, value, *, save=False):
        number = self.counts.get(name, 0)
        self.counts[name] = number + 1
        self.events.append({'name': name, 'number': number,
                            'fingerprints': fingerprints(value)})
        if save and number == 0:
            torch.save(cpu_tree(value), self.output / (name + '.pt'))


class RecordedRegion:
    def __init__(self, region, name, recorder):
        self.region, self.name, self.recorder = region, name, recorder

    def __getattr__(self, name):
        return getattr(self.region, name)

    def __call__(self, *args, **kwargs):
        self.recorder.record(self.name + '_input', (args, kwargs), save=True)
        result = self.region(*args, **kwargs)
        self.recorder.record(self.name + '_output', result, save=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('compile', 'eager'), default='compile')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--decisions', type=int, default=2)
    parser.add_argument('--reset-sequence', default='0,1,0')
    parser.add_argument('--no-record', action='store_true',
                        help='Run untouched production methods; retain actions and traces only.')
    args = parser.parse_args()
    assert 1 <= args.decisions <= 10, 'This is a bounded diagnostic, not an evaluation.'
    resets = [int(x) for x in args.reset_sequence.split(',')]
    assert all(x in (0, 1) for x in resets)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    run = manifest['runs'][0]
    checkpoint = Path(manifest['checkpoint']['path'])
    assert _file_sha256(checkpoint) == manifest['checkpoint']['sha256']
    resolved = {'algorithm_config': copy.deepcopy(run['config']),
                'environment': manifest['protocol']['environment'],
                'selector': run['selector']}
    params = resolved['algorithm_config']['alg_params']
    assert params['inner_rounds'] == 1
    params['compile'] = args.mode == 'compile'
    params['compile_strict'] = args.mode == 'compile'
    params['inner_replay_reset_each_round'] = bool(resets[0])
    env = _make_env(resolved)
    model = None
    try:
        model, _ = _initialize_frozen_model(resolved, env, checkpoint, 55, device=args.device)
        before = _outer_state_digest(model)
        engine = model.agent.inner_engine
        warm = env.reset(seed=101)[0]
        model.predict(warm, deterministic=True, episode_start=True)
        runtime = dict(python=platform.python_version(), torch=torch.__version__,
                       cuda=torch.version.cuda, hostname=platform.node(),
                       deterministic=torch.are_deterministic_algorithms_enabled(),
                       matmul_precision=torch.get_float32_matmul_precision(),
                       allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                       cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                       environment={k: os.environ.get(k) for k in (
                           'CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
                           'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'OMP_NUM_THREADS')})
        import evaluate_ambi_checkpoint
        runtime['evaluator_file'] = evaluate_ambi_checkpoint.__file__
        runtime['imported_source_commit'] = subprocess.check_output(
            ['git', '-C', str(Path(evaluate_ambi_checkpoint.__file__).parent), 'rev-parse', 'HEAD'],
            text=True).strip()
        if torch.cuda.is_available():
            runtime['gpu'] = subprocess.check_output([
                'nvidia-smi', '--query-gpu=name,uuid,driver_version',
                '--format=csv,noheader'], text=True).strip()
        report = dict(runtime=runtime, mode=args.mode, checkpoint_sha256=manifest['checkpoint']['sha256'],
                      manifest_code=manifest['code'], cases=[])
        original_regions = engine._compile_regions.copy()
        original_collect = engine._collect_round
        original_sample = engine._sample_batch
        original_clip = torch.nn.utils.clip_grad_norm_
        for repeat, reset in enumerate(resets):
            case_dir = args.output / f'case-{repeat}-reset-{reset}'
            case_dir.mkdir()
            rec = Recorder(case_dir)
            if args.no_record:
                rec.record = lambda *a, **kw: None
            engine.cfg.inner_replay_reset_each_round = bool(reset)
            engine.reset_for_evaluation(solver_seed(55, 'episode', 101), reuse_action_pool=True)
            engine._compile_regions = {k: RecordedRegion(v, k, rec) for k, v in original_regions.items()}

            def collect(root):
                state = engine.state
                rec.record('collect_initial', dict(root=root, actor=state.actor.state_dict(),
                    critic=state.critic.state_dict(), target=state.critic_target.state_dict(),
                    alpha=state.log_alpha, replay_size=state.replay.size), save=True)
                result = original_collect(root)
                rec.record('collected', state.replay.training_state_dict(), save=True)
                return result

            def sample(indices=None):
                result = original_sample(indices)
                rec.record('sample', result, save=True)
                return result

            def clip(parameters, *a, **kw):
                parameters = list(parameters)
                rec.record('grad', [p.grad for p in parameters], save=True)
                return original_clip(parameters, *a, **kw)

            engine._collect_round = collect
            engine._sample_batch = sample
            torch.nn.utils.clip_grad_norm_ = clip
            if args.no_record:
                engine._compile_regions = original_regions.copy()
                engine._collect_round, engine._sample_batch = original_collect, original_sample
                torch.nn.utils.clip_grad_norm_ = original_clip
            _seed_spaces(env, 101)
            obs, _ = env.reset(seed=101)
            decisions = []
            for step in range(args.decisions):
                rec.record('observation', torch.as_tensor(obs), save=True)
                rec.record('rng_before', engine.rng.training_state_dict())
                trace = InnerActionTrace(probes=True, probe_mode='outer_tail', probe_rollouts=32,
                    probe_horizon=int(model.cfg.inner_rollout_horizon),
                    probe_seed=solver_seed(55, 'togo_probe', 101, step))
                action, _ = model.predict(obs, deterministic=True, episode_start=step == 0, trace=trace)
                rec.record('rng_after', engine.rng.training_state_dict())
                rec.record('action', torch.as_tensor(action), save=True)
                events = copy.deepcopy(trace.events)
                for event in events:
                    event['metrics'] = {k: v for k, v in event.get('metrics', {}).items()
                                        if 'seconds' not in k}
                obs, reward, terminated, truncated, _ = env.step(action)
                decisions.append(dict(action=np.asarray(action).tolist(), reward=float(reward), trace=events))
                assert not (terminated or truncated)
            torch.nn.utils.clip_grad_norm_ = original_clip
            engine._collect_round, engine._sample_batch = original_collect, original_sample
            engine._compile_regions = original_regions.copy()
            assert _outer_state_digest(model) == before
            case = dict(repeat=repeat, reset=reset, decisions=decisions, fingerprints=rec.events)
            report['cases'].append(case)
            report['compiler_kernels'] = compiler_kernels(args.output / 'generated')
            (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f'Completed repeat={repeat} reset={reset}', flush=True)
    finally:
        if model is not None:
            model.close()
        env.close()


if __name__ == '__main__':
    main()
