#!/usr/bin/env python3
"""容忍 vLLM 返回的 NaN logprob,不让它杀死训练(幂等)。

问题 —— 2026-07-26 实测,Gemma3-12B 在 MM-UPT 配方下必崩:

    trl/trainer/grpo_trainer.py:1903
    sampling_per_token_logps = [torch.tensor(logps) for logps in sampling_per_token_logps_list]
    RuntimeError: Could not infer dtype of NoneType

根因不是"配置不兼容",是**数值溢出**:

  trl/generation/vllm_generation.py:88(原注释写着 "NaN logprob values are replaced with None")
      seq_logprobs.append([None if math.isnan(item.logprob) else item.logprob ...])

  → vLLM 算出 NaN 的 logprob → TRL 转成 None → torch.tensor(None) 抛异常 → 整个 run 挂掉。

为什么老配方能跑、新配方崩:
  老配方 temperature=1.0;MM-UPT 配方 temperature=0.7。
  温度是做除法的,0.7 相当于把 logits 放大 1.43 倍 —— Gemma3 的 logits 本来就大
  (它有 logit softcapping 机制),再放大就容易溢出成 NaN。
  这也解释了崩溃步数飘忽(实测 step 4 / 5 / 21):NaN 是随机出现的,取决于采样到哪些样本。

修法:把 None 替换成**该序列内有效 logprob 的均值**(缺失值用同分布的典型值填)。
  - 为什么不用 0.0:logprob=0 意味着概率 1,会让该 token 的 importance ratio 严重偏移
  - 用均值 → ratio ≈ 1(中性),而且 `vllm_importance_sampling_mode=token_truncate`
    还会把异常 ratio 截到 3.0,伤害有界
  - 整段序列全是 NaN 时(极罕见)退化为 0.0
  - 每次发生都打印计数,便于观察频率;频率高说明模型真的在数值上不稳定,要另外处理

作用文件:site-packages/trl/trainer/grpo_trainer.py
退出码:0=已打/本来就打过  1=没找到目标代码  2=文件不存在
"""
import io
import os
import re
import sys

MARKER = "PATCH-cogrpo-nanlogprob-v1"

OLD = """        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps) for logps in sampling_per_token_logps_list]"""

NEW = f'''        if sampling_per_token_logps_list is not None:
            # {MARKER}
            # vLLM 对溢出的 token 返回 NaN,TRL 的 extract_logprobs 把 NaN 转成 None,
            # 直接 torch.tensor(None) 会抛 RuntimeError 并杀死整个训练。
            # 用序列内有效值的均值填补(ratio≈1,中性),token_truncate 还会兜底截断。
            def _fill_nan_logps(_seq):
                if not any(_x is None for _x in _seq):
                    return _seq
                _valid = [_x for _x in _seq if _x is not None]
                _fill = (sum(_valid) / len(_valid)) if _valid else 0.0
                _n = len(_seq) - len(_valid)
                print(f"[nan-logprob] ⚠️ 本序列 {{_n}}/{{len(_seq)}} 个 token 的 logprob 为 NaN,"
                      f"用均值 {{_fill:.4f}} 填补(训练继续)", flush=True)
                return [_fill if _x is None else _x for _x in _seq]

            sampling_per_token_logps_list = [
                _fill_nan_logps(_s) for _s in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = [torch.tensor(logps) for logps in sampling_per_token_logps_list]'''


def patch(path: str) -> int:
    if not os.path.exists(path):
        print(f"[nan-logprob] ✗ 不存在: {path}")
        return 2
    with io.open(path, encoding="utf-8") as fh:
        src = fh.read()

    if MARKER in src:
        print(f"[nan-logprob] ✓ 已打过补丁: {path}")
        return 0
    if OLD not in src:
        print(f"[nan-logprob] ✗ 找不到目标代码(TRL 结构可能变了): {path}")
        return 1

    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(OLD, NEW, 1))
    print(f"[nan-logprob] ✓ 已修复: {path}")
    return 0


def main() -> int:
    import glob
    pats = glob.glob(os.environ.get("TRL_GRPO_GLOB", "/weka/scratch/jhu/dssg2026-ext-rghani1/yyang331/mllm-repro/env/lib/python3.*/site-packages/trl/trainer/grpo_trainer.py"))
    if not pats:
        print("[nan-logprob] ✗ 找不到 trl/trainer/grpo_trainer.py")
        return 2
    rc = 0
    for p in pats:
        rc = max(rc, patch(p))
    return rc


if __name__ == "__main__":
    sys.exit(main())
