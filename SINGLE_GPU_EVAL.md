# 单卡评测部署 · RTX PRO 6000 (Blackwell, sm_120)

面向一台 8×RTX PRO 6000 的机器,只用其中一张卡跑大档 Qwen2.5-VL-7B 那一列。
写于 2026-08-17。

## 一、先确认的三件事(不确认就别开跑)

### 1. vLLM 有没有 sm_120 的 kernel —— **最大风险**

预编译 wheel 大多只带 sm_80/sm_90。跑之前先测,**30 秒,不用等排队**:

```bash
python -c "
import torch, vllm
print('torch', torch.__version__, '| vllm', vllm.__version__)
print('device', torch.cuda.get_device_name(0), '| cap', torch.cuda.get_device_capability(0))
print('torch 支持的架构:', torch.cuda.get_arch_list())
"
```

- `get_arch_list()` 里有 `sm_120` → torch 侧 OK
- 没有 → 换 torch(cu128 的 2.9+ 通常带 sm_120),否则任何 kernel 都跑不了

vLLM 侧只能实跑才知道。用最小模型试一次:

```bash
python -c "
from vllm import LLM
llm = LLM(model='OpenGVLab/InternVL3_5-2B-HF', dtype='bfloat16',
          gpu_memory_utilization=0.5, max_model_len=2048,
          enforce_eager=True, trust_remote_code=True,
          limit_mm_per_prompt={'image': 1})
print('引擎启动成功')
"
```

起不来就换 vLLM 版本,别往下走。

### 2. `VLLM_WORKER_MULTIPROC_METHOD=spawn` —— **必须设**

vLLM 把推理引擎放在子进程里跑,而 **CUDA 不是 fork-safe**:父进程一旦初始化过 CUDA,
fork 出来的子进程再初始化就会失败,报

```
RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.
```

vLLM 0.11.2 的默认值是 `fork`(见 `vllm/envs.py`),所以**必须显式改成 spawn**。
`eval/run_bigtier_qwen.sh` 里已经 export 了;手工调 `eval_mllm.py` 时要自己设。

### 3. Blackwell 不需要 ViT 补丁 —— **省一步**

Qwen2.5-VL 的视觉塔 `head_dim=80`,不是 32 的倍数,FlashAttention 的 kernel 拒绝执行
(`headdim not being a multiple of 32`)。A100 上必须打补丁才能绕开。

**Blackwell 不用**:vLLM 在 `platforms/cuda.py` 里对 sm_100+ 直接强制 `TORCH_SDPA`:

```python
# For Blackwell GPUs, force TORCH_SDPA for now.
if cls.has_device_capability(100):
    return AttentionBackendEnum.TORCH_SDPA
```

`eval/eval_mllm.py` 还额外传了 `mm_encoder_attn_backend="TORCH_SDPA"` 作为双保险,
对非 Qwen 模型(InternVL head_dim=128 等)无影响。

## 二、准备

```bash
export HF_TOKEN=...                      # Gemma 系需要;Qwen/InternVL 公开
python eval/prepare_benchmarks.py all    # 五个 benchmark,行数硬校验
```

行数必须是:MathVision 3040 / MathVerse 3940 / MathVista 1000 /
We-Math 1740 / CoreCognition 1423。对不上就是构建失败,重跑那一个。

## 三、跑

```bash
CUDA_VISIBLE_DEVICES=2 bash eval/run_bigtier_qwen.sh \
    --out_root work_dirs/eval_bigtier
```

先用小样本验证管线通(约 10 分钟):

```bash
CUDA_VISIBLE_DEVICES=2 bash eval/run_bigtier_qwen.sh \
    --out_root work_dirs/eval_smoke --limit 16
```

`--gt_ckpt <path>` 可加上第四格(mmupt 重训的 GT);不给就只跑前三格。

**断点续跑**:某个 bench 的 json 已完整就跳过,崩了原样重跑即可。

## 四、口径(冻结,改了整表作废)

| 项 | 值 |
|---|---|
| temperature | **0**(贪心) |
| max_tokens | **16384** |
| max_model_len | **24576**(保证 16k 是真预算,不被上下文压顶) |
| prompt | **全表 boxed**,含训练过的 ckpt |
| 判分 | **纯规则**,不用 LLM judge |
| benchmark | 五个,报 **AVG5** |
| 选点 | best-by-val |

判分保留了一处相对上游的修正:**选择题同时接受选项字母和该选项的值**
(标准答案是 `D`、模型答 `90` 也算对)。去掉这条,答数值的模型会凭空掉 20+ 分,
是一把偏袒回答风格的尺子。

## 五、四个格子

| tag | ckpt |
|---|---|
| `base-qwenvl7b` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `ttrl-q7b-mmupt` | `q1716523669/mllm-mmr1-ttrl-qwen25vl7b-mmupt-full/best` |
| `co-q7b-x-i8b` | `q1716523669/mllm-cogrpo-heter-qwen25vl-7b-x-internvl35-8b-mmr1-mmupt-groupA-qwen25vl-7b` |
| `gt-q7b-mmupt` | 需 `--gt_ckpt` 指定(mmupt 重训产物) |

`co` 选 `×InternVL-8B` 这一对,是为了和同列 TTRL 的训练 cap 对齐(都是 2048)。

## 六、耗时

A100 双卡实测每格 2–7 小时。Blackwell 单卡:算力更强但少一张卡,大致抵消,
**每格约 2.5–5 小时,四格串行 12–18 小时**。

96 GB 显存跑 7B 绰绰有余,可以起两个进程各跑一半 benchmark 把时间压到 6–9 小时:

```bash
ONLY="mathvision wemath"                CUDA_VISIBLE_DEVICES=2 bash eval/run_eval_all.sh ... &
ONLY="mathverse mathvista corecognition" CUDA_VISIBLE_DEVICES=3 bash eval/run_eval_all.sh ... &
wait
```

## 七、拓扑提醒(多卡训练时才相关)

这台机器没有 NVLink,全 PCIe。`GPU0-3` 在 NUMA 节点 0,`GPU4-7` 在节点 1,
跨组走最慢的 SYS 路径。**多卡训练要把卡选在同一组内**(0-3 或 4-7),
不要像 `3,4,5,6` 那样横跨——ZeRO-3 每层都要 all-gather,对互联带宽敏感。
开跑前先 `nvidia-smi topo -m`。

单卡评测不受影响。
