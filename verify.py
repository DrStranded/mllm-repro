"""1-shot install verifier for the `mllm-repro` env (FROZEN stack B).

Stack B is the PROVEN Anvil `mllm-cogrpodp-v2` environment, captured verbatim.
Do NOT target torch2.10 / vllm0.18 / transformers5.x — those are the old
aspirational ENV.md numbers and are NOT what this repo runs.

Expected versions (frozen stack B):
    torch         2.9.0+cu128    (cu128 runtime)
    trl           1.5.0.dev0     (DrStranded fork, editable)
    transformers  4.57.0
    vllm          0.11.2
    deepspeed     0.18.0
    flash_attn    2.8.3          (cu128 / torch2.9, cxx11abi FALSE)

MLLM extras (should be in the frozen venv):
    qwen-vl-utils 0.0.14, opencv-python-headless, timm, av

Grader path (qwen-sympy via 2-hop wrapper, NOT math-verify):
    co_label_utils.py:25 -> verifiers/math_verify_wrapper.py:44 -> verifiers/qwen/math_grade.py

Two kinds of check:
    kind="pure"  — pure-python (versions / registry / extras / grader). Safe to run
                   at DOCKER BUILD time as a build-gate. Guards every optional import
                   so a missing dep is reported, never an uncaught crash.
    kind="gpu"   — RUNTIME-ONLY (needs a live GPU): cuda bf16 matmul, flash-attn forward.

Run modes:
    python verify.py            # all 6 checks (runtime; needs GPU)
    python verify.py --pure     # only the 4 pure-python checks (build-gate; no GPU)
    VERIFY_MODE=build python verify.py   # same as --pure
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import traceback

PASS_MARK = "✅"
FAIL_MARK = "❌"
# each entry: (name, fn, kind)  where kind ∈ {"pure", "gpu"}
CHECKS: list[tuple[str, callable, str]] = []


def check(name: str, kind: str = "pure"):
    def deco(fn):
        CHECKS.append((name, fn, kind))
        return fn
    return deco


def _dist_version(dist_name: str) -> str:
    """Distribution version via stdlib importlib.metadata (no pkg_resources)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(dist_name)
    except Exception:
        return "unknown"


# ----- Check 1: torch + CUDA bf16 matmul (RUNTIME-ONLY) -----
@check("[1/6] torch + CUDA (runtime)", kind="gpu")
def _torch_cuda():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available()=False — driver/CUDA 装错"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    runtime = torch.version.cuda
    print(f"   torch {torch.__version__}, cuda runtime {runtime}, GPU '{name}' cap {cap}")
    assert runtime.startswith("12.8") or runtime.startswith("12.9"), (
        f"cuda runtime {runtime} 不是 12.8 — 冻结栈 B 是 torch2.9.0+cu128"
    )
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    assert y.shape == (64, 64)
    print(f"   bf16 matmul OK on {name}")


# ----- Check 2: locked version table (pure) -----
@check("[2/6] 冻结栈版本表(stack B)", kind="pure")
def _versions():
    # (import name, expected-prefix). trl/torch/flash_attn 会带 +local / .devN 后缀。
    specs = [
        ("torch",        "2.9.0"),
        ("trl",          "1.5.0"),
        ("transformers", "4.57.0"),
        ("vllm",         "0.11.2"),
        ("deepspeed",    "0.18.0"),
        ("flash_attn",   "2.8.3"),
    ]
    bad = []
    for pkg, want in specs:
        try:
            mod = importlib.import_module(pkg)
            raw = getattr(mod, "__version__", None) or _dist_version(pkg)
            got = str(raw).split("+")[0].split(".dev")[0]
            ok = got.startswith(want)
        except Exception as e:
            got = f"<import failed: {type(e).__name__}: {e}>"
            ok = False
        mark = PASS_MARK if ok else FAIL_MARK
        print(f"   {mark} {pkg:14s} got {got:26s} want {want}")
        if not ok:
            bad.append(pkg)
    if bad:
        raise AssertionError(
            f"version mismatch / 缺失 on {bad} —— 不是冻结栈 B。"
            f" 期望 torch2.9.0 / transformers4.57.0 / vllm0.11.2 / deepspeed0.18.0"
            f" / flash-attn2.8.3 / trl1.5.0。"
            f" 严禁 transformers 5.x / vllm 0.18+ / flash-attn cxx11abi TRUE。"
        )


# ----- Check 3: flash-attn forward (ABI sanity, RUNTIME-ONLY) -----
@check("[3/6] flash-attn forward(ABI, runtime)", kind="gpu")
def _flash_attn_forward():
    import torch
    from flash_attn import flash_attn_func
    q = k = v = torch.randn(1, 4, 1, 64, dtype=torch.float16, device="cuda")
    out = flash_attn_func(q, k, v)
    assert out.shape == q.shape
    print(f"   flash_attn_func OK, out shape {tuple(out.shape)}")


# ----- Check 4: MLLM extras present (pure) -----
@check("[4/6] MLLM 杂项保留(stack B)", kind="pure")
def _mllm_extras():
    missing = []
    for import_name, dist_name in [
        ("qwen_vl_utils", "qwen-vl-utils"),
        ("cv2",           "opencv-python-headless"),
        ("timm",          "timm"),
        ("av",            "av"),
    ]:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", None) or _dist_version(dist_name)
            print(f"   {PASS_MARK} {dist_name:24s} {ver}")
        except Exception as e:
            print(f"   {FAIL_MARK} {dist_name:24s} import 失败: {type(e).__name__}: {e}")
            missing.append(dist_name)

    # math-verify MUST NOT be present (grader 走 qwen-sympy 2-hop wrapper)
    try:
        importlib.import_module("math_verify")
        raise AssertionError(
            "math-verify 装着!grader 走 qwen-sympy wrapper,math-verify 是 dead weight,"
            "更糟糕的是它会拉新 antlr4 把 latex2sympy2 1.9.x 路径炸了。"
            "卸:`pip uninstall -y math-verify latex2sympy2_extended`"
        )
    except ImportError:
        print(f"   {PASS_MARK} math-verify 未装(预期)")

    if missing:
        raise AssertionError(
            f"缺 MLLM extras {missing} —— 冻结栈 B 应全部保留;检查 constraints.txt 是否装全。"
        )


# ----- Check 5: AutoConfig registry has our MLLM model types (pure) -----
@check("[5/6] AutoConfig registry(transformers 4.57.0)", kind="pure")
def _autoconfig_registry():
    from transformers import CONFIG_MAPPING

    needed = ["qwen2_5_vl", "internvl", "gemma3"]
    missing = [m for m in needed if m not in CONFIG_MAPPING]
    for m in needed:
        mark = PASS_MARK if m in CONFIG_MAPPING else FAIL_MARK
        print(f"   {mark} model_type '{m}'")
    if missing:
        raise AssertionError(
            f"transformers 4.57.0 缺 {missing} —— 应全部 register。核对 transformers 版本。"
        )

    # NOTE: qwen3_vl / qwen3_vl_moe 之前被误当成 "transformers 5.x-only",
    #       但它们在 4.57.x 里就已存在 —— 对它们断言"必须缺席"会在冻结栈
    #       tf4.57.0 上误崩(build-gate ABORT)。故不再断言其缺席,存在即报告。
    for m in ["qwen3_vl", "qwen3_vl_moe"]:
        if m in CONFIG_MAPPING:
            print(f"   ℹ️  model_type '{m}' 存在(4.57.x 起自带,非错误)")

    # 真·transformers-5.x-only 的 model_type(如 gemma4)不该出现;
    # 出现只当作 WARNING(说明装了 5.x),不 assert,免得炸 build-gate。
    future_only = ["gemma4"]
    leaked = [m for m in future_only if m in CONFIG_MAPPING]
    if leaked:
        print(f"   ⚠️  registry 里出现 {leaked} —— 这是 transformers 5.x 才加的;"
              f"冻结栈应为 4.57.0,跑 `pip show transformers` 核对版本。")
    else:
        print(f"   {PASS_MARK} 未见 5.x-only model_type {future_only}(预期)")


# ----- Check 6: grader 2-hop callable + trl EOS patch (pure) -----
@check("[6/6] grader 2-hop(qwen-sympy)+ trl EOS patch", kind="pure")
def _grader_and_eos_patch():
    # 6a — trl GRPOTrainer EOS patch present (Phi-3.5 / Qwen3 list-eos 修复)
    #      trl import 失败不该炸整条 check(纯 python build-gate 里 trl 可能未装)。
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
        src = inspect.getsource(GRPOTrainer.__init__)
        if "self.eos_token_ids" not in src:
            print(f"   ⚠️  trl GRPOTrainer 没有 self.eos_token_ids EOS patch")
            print(f"   ⚠️  Phi-3.5 / Qwen3 list-typed eos_token_id 会假报 clipped_ratio=0.97")
            print(f"   ⚠️  这个 patch 在 LLM 仓 trl/ 里;DrStranded/trl fork 还没同步。"
                  "只跑 MLLM(Qwen2.5-VL/InternVL/Gemma-3)不阻塞;LLM 共用时要 sync。")
        else:
            print(f"   {PASS_MARK} trl EOS patch detected")
    except Exception as e:
        print(f"   ⚠️  trl 检查跳过(import 失败: {type(e).__name__}: {e})")

    # 6b — qwen-sympy 2-hop grader callable
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainers"),
        os.path.expanduser("~/research/trl-projects-mllm/trainers"),
    ]
    wrapper_seen = False
    last_err = None
    for p in candidate_paths:
        wrapper = os.path.join(p, "verifiers", "math_verify_wrapper.py")
        if os.path.isfile(wrapper):
            wrapper_seen = True
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                from verifiers.math_verify_wrapper import grade_answer
                # 含 Unicode `°` 的 case(wrapper 价值所在)
                ok_unicode = grade_answer("90°", "90")
                # LaTeX 等价(qwen-sympy 价值所在)
                ok_latex = grade_answer("1/2", "\\frac{1}{2}")
                assert ok_unicode and ok_latex, (
                    f"grader 异常:Unicode '90°' vs '90' = {ok_unicode}, "
                    f"LaTeX '1/2' vs '\\frac{{1}}{{2}}' = {ok_latex}"
                )
                print(f"   {PASS_MARK} qwen-sympy 2-hop grader callable")
                print(f"     - grade('90°', '90') = True(wrapper Unicode strip)")
                print(f"     - grade('1/2', '\\\\frac{{1}}{{2}}') = True(qwen-sympy LaTeX equiv)")
                return
            except Exception as e:
                last_err = e
                print(f"   ⚠️  在 {p} 试 grader 失败: {type(e).__name__}: {e}")

    if wrapper_seen:
        raise AssertionError(
            f"找到 grader wrapper 但调用失败(常见:缺 pylatexenc / latex2sympy2 / "
            f"antlr4-python3-runtime,应在 constraints.txt 里)。最后错误: {last_err}"
        )
    raise AssertionError(
        "找不到 grader wrapper(trainers/verifiers/math_verify_wrapper.py)。"
        "你不在 mllm-repro 仓里跑,或 trainers/ 缺 verifiers/ 目录。"
    )


# ----- Run -----
def main(pure_only: bool = False) -> int:
    selected = [(n, f, k) for (n, f, k) in CHECKS if (k == "pure" or not pure_only)]
    mode = "PURE (build-gate, no GPU)" if pure_only else "FULL (runtime, needs GPU)"
    print("=" * 72)
    print(f"  mllm-repro verify.py — frozen stack B — mode: {mode}")
    print("=" * 72)
    fail = 0
    for name, fn, kind in selected:
        print(f"\n{name}")
        try:
            fn()
            print(f"   {PASS_MARK} pass")
        except Exception as e:
            fail += 1
            print(f"   {FAIL_MARK} {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print()
    print("=" * 72)
    if fail == 0:
        print(f"  {PASS_MARK} ALL {len(selected)} CHECKS PASSED")
        print("=" * 72)
        print("frozen stack B baseline verified. Ready for MLLM training/eval.")
        return 0
    else:
        print(f"  {FAIL_MARK} {fail}/{len(selected)} CHECKS FAILED")
        print("=" * 72)
        return 1


if __name__ == "__main__":
    _pure = (
        "--pure" in sys.argv
        or os.environ.get("VERIFY_MODE", "").lower() in ("build", "pure")
        or os.environ.get("VERIFY_PURE", "") == "1"
    )
    sys.exit(main(pure_only=_pure))
