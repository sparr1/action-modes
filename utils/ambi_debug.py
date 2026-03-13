import torch
import numpy as np
from typing import Any, Mapping, Iterable
import mujoco


def _snapshot_policy_tensors(policy):
    """
    Snapshot all policy params + buffers from state_dict().
    Returns CPU clones so we can compare before/after train().
    """
    out = {}
    for k, v in policy.state_dict().items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
    return out


def _policy_tensor_meta(policy):
    """
    Metadata for each named parameter/buffer.
    """
    meta = {}

    for name, p in policy.named_parameters():
        meta[name] = {
            "kind": "param",
            "requires_grad": bool(p.requires_grad),
            "shape": tuple(p.shape),
        }

    for name, b in policy.named_buffers():
        if name not in meta:
            meta[name] = {
                "kind": "buffer",
                "requires_grad": False,
                "shape": tuple(b.shape),
            }

    return meta


def _module_name_from_key(k: str) -> str:
    # e.g. actor.latent_pi.0.lora_A -> actor.latent_pi.0
    #      actor.latent_pi.0.base.weight -> actor.latent_pi.0.base
    return k.rsplit(".", 1)[0] if "." in k else "<root>"


def print_policy_update_report(
    policy,
    before_snapshot,
    *,
    tag="",
    atol=0.0,
    rtol=0.0,
    ignore_prefixes=(),
    max_modules_to_print=200,
):
    """
    Compare a previous snapshot against current policy state and print:
      - which modules changed
      - which modules stayed the same
      - which changed tensors were trainable vs frozen
      - whether any unexpected frozen tensors moved

    Returns a dict summary.
    """
    after_snapshot = _snapshot_policy_tensors(policy)
    meta = _policy_tensor_meta(policy)

    ignore_prefixes = tuple(ignore_prefixes)

    changed_by_module = {}
    same_by_module = {}

    changed_trainable = []
    changed_frozen = []
    changed_frozen_non_target = []
    changed_frozen_target = []

    all_keys = sorted(set(before_snapshot.keys()) | set(after_snapshot.keys()))

    for k in all_keys:
        if any(k.startswith(prefix) for prefix in ignore_prefixes):
            continue

        if k not in before_snapshot or k not in after_snapshot:
            module = _module_name_from_key(k)
            changed_by_module.setdefault(module, []).append({
                "name": k,
                "status": "missing_before_after_mismatch",
                "requires_grad": meta.get(k, {}).get("requires_grad", False),
                "kind": meta.get(k, {}).get("kind", "unknown"),
            })
            continue

        a = before_snapshot[k]
        b = after_snapshot[k]

        if a.shape != b.shape:
            module = _module_name_from_key(k)
            changed_by_module.setdefault(module, []).append({
                "name": k,
                "status": "shape_changed",
                "shape_before": tuple(a.shape),
                "shape_after": tuple(b.shape),
                "requires_grad": meta.get(k, {}).get("requires_grad", False),
                "kind": meta.get(k, {}).get("kind", "unknown"),
            })
            continue

        same = torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        module = _module_name_from_key(k)

        if same:
            same_by_module.setdefault(module, []).append(k)
        else:
            diff = (a - b).abs()
            max_diff = diff.max().item()
            flat_idx = int(diff.view(-1).argmax().item())

            item = {
                "name": k,
                "status": "changed",
                "max_abs_diff": max_diff,
                "flat_idx": flat_idx,
                "requires_grad": meta.get(k, {}).get("requires_grad", False),
                "kind": meta.get(k, {}).get("kind", "unknown"),
            }
            changed_by_module.setdefault(module, []).append(item)

            if item["requires_grad"]:
                changed_trainable.append(k)
            else:
                changed_frozen.append(k)
                if k.startswith("critic_target."):
                    changed_frozen_target.append(k)
                else:
                    changed_frozen_non_target.append(k)

    changed_modules = sorted(changed_by_module.keys())
    same_modules = sorted(set(same_by_module.keys()) - set(changed_by_module.keys()))

    header = f"[inner-debug] {tag}" if tag else "[inner-debug]"
    print("\n" + "=" * 100)
    print(header)
    print(
        f"changed_modules={len(changed_modules)} | "
        f"same_modules={len(same_modules)} | "
        f"changed_trainable_tensors={len(changed_trainable)} | "
        f"changed_frozen_tensors={len(changed_frozen)}"
    )

    if changed_frozen_non_target:
        print("UNEXPECTED frozen tensors changed (these should be inspected):")
        for name in changed_frozen_non_target:
            print(f"  - {name}")

    if changed_frozen_target:
        print("Frozen critic_target tensors changed (usually expected from Polyak update):")
        for name in changed_frozen_target:
            print(f"  - {name}")

    print("\nModules that CHANGED:")
    for module_name in changed_modules[:max_modules_to_print]:
        items = changed_by_module[module_name]
        print(f"  {module_name}")
        for item in items:
            short_name = item["name"].split(".")[-1]
            if item["status"] == "changed":
                print(
                    f"    changed: {short_name} | "
                    f"kind={item['kind']} | "
                    f"requires_grad={item['requires_grad']} | "
                    f"max_abs_diff={item['max_abs_diff']:.3e} | "
                    f"flat_idx={item['flat_idx']}"
                )
            elif item["status"] == "shape_changed":
                print(
                    f"    shape_changed: {short_name} | "
                    f"{item['shape_before']} -> {item['shape_after']}"
                )
            else:
                print(f"    {item['status']}: {short_name}")

        same_here = same_by_module.get(module_name, [])
        if same_here:
            same_short = [x.split(".")[-1] for x in same_here]
            print(f"    same: {same_short}")

    if len(changed_modules) > max_modules_to_print:
        print(f"  ... truncated {len(changed_modules) - max_modules_to_print} more changed modules")

    print("\nModules that stayed EXACTLY THE SAME:")
    for module_name in same_modules[:max_modules_to_print]:
        short = [x.split(".")[-1] for x in same_by_module[module_name]]
        print(f"  {module_name}: {short}")

    if len(same_modules) > max_modules_to_print:
        print(f"  ... truncated {len(same_modules) - max_modules_to_print} more same modules")

    print("=" * 100 + "\n")

    return {
        "changed_modules": changed_modules,
        "same_modules": same_modules,
        "changed_trainable": changed_trainable,
        "changed_frozen": changed_frozen,
        "changed_frozen_non_target": changed_frozen_non_target,
        "changed_frozen_target": changed_frozen_target,
        "changed_by_module": changed_by_module,
        "same_by_module": same_by_module,
    }

def state_dicts_equal(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    exact: bool = False,
    ignore_keys: Iterable[str] = (),
) -> bool:
    ignore = set(ignore_keys)
    a_keys = set(a.keys()) - ignore
    b_keys = set(b.keys()) - ignore
    if a_keys != b_keys:
        return False

    for k in a_keys:
        va, vb = a[k], b[k]
        if torch.is_tensor(va) and torch.is_tensor(vb):
            xa = va.detach()
            xb = vb.detach()
            if xa.shape != xb.shape or xa.dtype != xb.dtype:
                return False
            if exact:
                if not torch.equal(xa, xb):
                    return False
            else:
                if not torch.allclose(xa, xb, rtol=rtol, atol=atol, equal_nan=True):
                    return False
        else:
            if va != vb:
                return False
    return True

def assert_sb3_weights_copied(
    outer_agent,
    inner_agent,
    obs_sample=None,
    *,
    check_action=False,
    deterministic=True,
    atol=0.0,
    rtol=0.0,
):
    def _unwrap_sb3_algo(agent):
        # Handles your Baseline wrapper and raw SB3 objects
        return agent.model if hasattr(agent, "model") else agent

    def _get_policy(agent):
        algo = _unwrap_sb3_algo(agent)
        if not hasattr(algo, "policy"):
            raise TypeError(
                f"Expected an SB3 algorithm with `.policy`, got: {type(algo)}"
            )
        return algo.policy

    outer_algo = _unwrap_sb3_algo(outer_agent)
    inner_algo = _unwrap_sb3_algo(inner_agent)
    outer_policy = _get_policy(outer_agent)
    inner_policy = _get_policy(inner_agent)

    errs = []

    # 1) Compare policy state_dict keys
    sd_out = outer_policy.state_dict()
    sd_in = inner_policy.state_dict()

    keys_out = list(sd_out.keys())
    keys_in = list(sd_in.keys())

    if keys_out != keys_in:
        missing_in = [k for k in keys_out if k not in sd_in]
        extra_in = [k for k in keys_in if k not in sd_out]
        raise AssertionError(
            "Policy state_dict key mismatch.\n"
            f"Missing in inner: {missing_in[:20]}\n"
            f"Extra in inner: {extra_in[:20]}"
        )

    # 2) Compare each tensor/buffer
    for k in keys_out:
        a = sd_out[k].detach().cpu()
        b = sd_in[k].detach().cpu()

        if a.shape != b.shape:
            errs.append(f"{k}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            continue

        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a - b).abs()
            max_diff = diff.max().item()
            idx = int(diff.view(-1).argmax().item())
            av = a.view(-1)[idx].item()
            bv = b.view(-1)[idx].item()
            errs.append(
                f"{k}: max_abs_diff={max_diff:.3e} at flat_idx={idx} "
                f"(outer={av:.9g}, inner={bv:.9g})"
            )

    # 3) Optional: compare predicted actions on same obs
    if check_action:
        if obs_sample is None:
            raise ValueError("obs_sample must be provided when check_action=True")

        # convert torch obs to numpy if needed
        if torch.is_tensor(obs_sample):
            obs_np = obs_sample.detach().cpu().numpy()
        else:
            obs_np = np.array(obs_sample, copy=False)

        # IMPORTANT: call predict on the SB3 algo object, not your wrapper (wrapper may not accept kwargs)
        act_out, _ = outer_algo.predict(obs_np, deterministic=deterministic)
        act_in, _ = inner_algo.predict(obs_np, deterministic=deterministic)

        if not np.allclose(act_out, act_in, atol=max(atol, 1e-7), rtol=max(rtol, 1e-6)):
            d = np.abs(act_out - act_in)
            idx = int(np.argmax(d))
            errs.append(
                "predict() action mismatch: "
                f"max_abs_diff={d.flat[idx]:.3e} at flat_idx={idx} "
                f"(outer={act_out.flat[idx]:.9g}, inner={act_in.flat[idx]:.9g})"
            )

    if errs:
        raise AssertionError(
            "Outer/inner SB3 weights do NOT match after copy:\n- " + "\n- ".join(errs[:50])
        )

    return True

def _debug_wrapper_type_chain(env):
    """Return wrapper -> ... -> base env type names."""
    types = []
    cur = env
    seen = set()
    while True:
        types.append(type(cur).__name__)
        if not hasattr(cur, "env"):
            break
        nxt = cur.env
        if id(nxt) in seen:  # just in case of a weird cycle
            types.append("<cycle>")
            break
        seen.add(id(nxt))
        cur = nxt
    return types


def _debug_get_wrapper_attr(env, name, default=None):
    try:
        if hasattr(env, "get_wrapper_attr"):
            return env.get_wrapper_attr(name)
    except Exception:
        pass
    return default


def _debug_get_full_mujoco_state(env):
    """Flattened MuJoCo FULLPHYSICS state as float64."""
    uw = env.unwrapped
    model, data = uw.model, uw.data
    spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    n = mujoco.mj_stateSize(model, spec)
    x = np.empty(n, dtype=np.float64)
    mujoco.mj_getState(model, data, x, spec)
    return x


def assert_envs_match_after_copy(
    outer_env,
    inner_env,
    outer_obs=None,
    *,
    check_wrapper_stack=False,
    check_outer_obs_vs_inner_raw=False,
    atol=1e-6,
    rtol=1e-5,
):
    errs = []

    if check_wrapper_stack:
        outer_chain = _debug_wrapper_type_chain(outer_env)
        inner_chain = _debug_wrapper_type_chain(inner_env)
        if outer_chain != inner_chain:
            errs.append(
                "Wrapper stack mismatch:\n"
                f"  outer={outer_chain}\n"
                f"  inner={inner_chain}"
            )

    try:
        if outer_env.observation_space.shape != inner_env.observation_space.shape:
            errs.append(
                f"Observation space shape mismatch: "
                f"{outer_env.observation_space.shape} vs {inner_env.observation_space.shape}"
            )
    except Exception as e:
        errs.append(f"Could not compare observation spaces: {e}")

    try:
        if outer_env.action_space.shape != inner_env.action_space.shape:
            errs.append(
                f"Action space shape mismatch: "
                f"{outer_env.action_space.shape} vs {inner_env.action_space.shape}"
            )
    except Exception as e:
        errs.append(f"Could not compare action spaces: {e}")

    try:
        s_outer = _debug_get_full_mujoco_state(outer_env)
        s_inner = _debug_get_full_mujoco_state(inner_env)
        if s_outer.shape != s_inner.shape:
            errs.append(f"MuJoCo state shape mismatch: {s_outer.shape} vs {s_inner.shape}")
        else:
            if not np.allclose(s_outer, s_inner, atol=atol, rtol=rtol):
                diff = np.abs(s_outer - s_inner)
                idx = int(np.argmax(diff))
                errs.append(
                    "MuJoCo FULLPHYSICS mismatch: "
                    f"max_abs_diff={diff[idx]:.3e} at index {idx} "
                    f"(outer={s_outer[idx]:.9g}, inner={s_inner[idx]:.9g})"
                )
    except Exception as e:
        errs.append(f"Could not compare MuJoCo full state: {e}")

    try:
        qo = np.array(outer_env.unwrapped.data.qpos, copy=True)
        qi = np.array(inner_env.unwrapped.data.qpos, copy=True)
        if not np.allclose(qo, qi, atol=atol, rtol=rtol):
            d = np.abs(qo - qi)
            idx = int(np.argmax(d))
            errs.append(
                f"qpos mismatch: max_abs_diff={d[idx]:.3e} at index {idx} "
                f"(outer={qo[idx]:.9g}, inner={qi[idx]:.9g})"
            )
    except Exception as e:
        errs.append(f"Could not compare qpos: {e}")

    try:
        vo = np.array(outer_env.unwrapped.data.qvel, copy=True)
        vi = np.array(inner_env.unwrapped.data.qvel, copy=True)
        if not np.allclose(vo, vi, atol=atol, rtol=rtol):
            d = np.abs(vo - vi)
            idx = int(np.argmax(d))
            errs.append(
                f"qvel mismatch: max_abs_diff={d[idx]:.3e} at index {idx} "
                f"(outer={vo[idx]:.9g}, inner={vi[idx]:.9g})"
            )
    except Exception as e:
        errs.append(f"Could not compare qvel: {e}")

    for k in ("_elapsed_steps", "_has_reset", "checked_step", "checked_reset", "checked_render"):
        ov = _debug_get_wrapper_attr(outer_env, k, default="<missing>")
        iv = _debug_get_wrapper_attr(inner_env, k, default="<missing>")
        if ov != iv:
            errs.append(f"Wrapper attr mismatch for {k}: outer={ov!r}, inner={iv!r}")

    try:
        if hasattr(outer_env.unwrapped, "_get_obs") and hasattr(inner_env.unwrapped, "_get_obs"):
            raw_outer = np.array(outer_env.unwrapped._get_obs(), copy=True)
            raw_inner = np.array(inner_env.unwrapped._get_obs(), copy=True)
            if raw_outer.shape != raw_inner.shape:
                errs.append(f"raw _get_obs shape mismatch: {raw_outer.shape} vs {raw_inner.shape}")
            elif not np.allclose(raw_outer, raw_inner, atol=atol, rtol=rtol):
                d = np.abs(raw_outer - raw_inner)
                idx = int(np.argmax(d))
                errs.append(
                    f"raw _get_obs mismatch: max_abs_diff={d[idx]:.3e} at index {idx} "
                    f"(outer={raw_outer[idx]:.9g}, inner={raw_inner[idx]:.9g})"
                )

            if outer_obs is not None and check_outer_obs_vs_inner_raw:
                outer_obs_arr = np.array(outer_obs, copy=False)
                if outer_obs_arr.shape != raw_inner.shape:
                    errs.append(
                        f"outer_obs vs inner raw obs shape mismatch: "
                        f"{outer_obs_arr.shape} vs {raw_inner.shape}"
                    )
                elif not np.allclose(outer_obs_arr, raw_inner, atol=atol, rtol=rtol):
                    d = np.abs(outer_obs_arr - raw_inner)
                    idx = int(np.argmax(d))
                    errs.append(
                        f"outer_obs vs inner raw obs mismatch: max_abs_diff={d[idx]:.3e} at index {idx} "
                        f"(outer_obs={outer_obs_arr[idx]:.9g}, inner_raw={raw_inner[idx]:.9g})"
                    )
    except Exception as e:
        errs.append(f"Could not compare raw observations: {e}")

    if errs:
        raise AssertionError("Env copy mismatch after _set_env_state:\n- " + "\n- ".join(errs))