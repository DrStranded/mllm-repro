# 8 卡节点并行运行两个 MLLM RL 实验

## 目标

在同一台 8 卡节点上并行运行两个 MLLM RL 实验：

- 实验 A 使用 GPU 0–3。
- 实验 B 使用 GPU 4–7。
- 两个实验均使用 `InternVL3.5-2B`、`mmr1` 数据集。
- 每个实验训练 `1 epoch`，预期共 `722 steps`。

---

## 1. 获取并检查仓库

```bash
git clone git@github.com:DrStranded/mllm-repro.git
cd mllm-repro
git log --oneline -1
```

检查最新提交：必须包含 `3e57239` 或为更新版本，否则 Gemma 会崩溃。

---

## 2. 安装环境

严格按照仓库 `README §3` 安装 frozen stack，不要更改版本：

| 组件 | 固定版本 |
|---|---|
| `torch` | `2.9.0+cu128` |
| `vllm` | `0.11.2` |
| `transformers` | `4.57.0` |
| `deepspeed` | `0.18.0` |
| `flash-attn` | `2.8.3 (cu12torch2.9)` |
| `trl` | `DrStranded/trl@9881fe1e` |

> [!IMPORTANT]
> `trl` 必须使用上述 fork 和 commit。

安装依赖时必须使用 pip constraints 文件锁定 `torch` 版本，否则 `xgrammar` 可能把 `torch` 替换成 `2.13`，破坏整个依赖栈。

---

## 3. 准备训练数据

### 方案一：准备全部数据

```bash
bash setup/prepare_data.sh
```

### 方案二：仅预处理 MMR1

先设置预处理数据根目录：

```bash
export PRE_ROOT=<预处理数据根目录>
```

然后运行：

```bash
python tools/preprocess_mllm_dataset.py \
  "MMR1/MMR1-Math-RL-Data-v0" \
  "$PRE_ROOT/mmr1_8k"
```

预期得到 `5782` 条数据。请在继续之前核对数量。

---

## 4. 准备 MathVista in-loop eval 数据

仓库中的 `data/mathvista/` 只有 JSONL，未提交 `images/`。in-loop eval 需要图片。

从 Hugging Face 数据集 `AI4Math/MathVista` 的 `testmini` split 中，按照 `problem` 文本匹配记录，并将图片导出为：

```text
images/N.png
```

共应导出 `150` 张图片。

推荐的最终结构：

```text
mllm-repro/
└── data/
    └── mathvista/
        ├── testmini_150.jsonl
        └── images/
            ├── 1.png
            ├── 2.png
            └── ...
```

如果暂时不做 MathVista 图片导出，也可以将 `MLLM_EVAL_PATH` 指向一个 dummy JSONL，使训练能够启动；但这样会导致 best-checkpoint 选择失效。

---

## 5. 实验配置

| 实验 | GPU | 启动脚本 |
|---|---:|---|
| A：GT | 0–3 | `examples/openr1_internvl35_2b_gt.sh` |
| B：TTRL | 4–7 | `examples/openr1_internvl35_2b_ttrl.sh` |

共同配置：

- 模型：`InternVL3.5-2B`
- 数据集：`mmr1`
- 训练轮数：`1 epoch`
- 预期训练步数：`722 steps`
- 每卡 batch size：`BS=2`
- gradient accumulation：`GA=8`
- 每个实验进程数：`NUM_PROC=4`

---

## 6. 设置共同环境变量

在仓库根目录中执行。请先替换所有 `<...>` 占位符：

```bash
export PRE_ROOT=<预处理数据根目录>
export MLLM_REPO=<mllm-repro 仓库绝对路径>

export MLLM_ENV_READY=1
export MLLM_PRE_DIR="$PRE_ROOT/mmr1_8k"
export MLLM_EVAL_PATH="$MLLM_REPO/data/mathvista/testmini_150.jsonl"
export MLLM_EVAL_IMAGE_DIR="$MLLM_REPO/data/mathvista/images"
export HF_TOKEN=<你的 Hugging Face token>
```

> [!IMPORTANT]
> `MLLM_PRE_DIR=$PRE_ROOT/mmr1_8k` 决定训练使用 `mmr1`，而不是 `openr1`。

建议在启动前确认变量：

```bash
printf 'MLLM_PRE_DIR=%s\n' "$MLLM_PRE_DIR"
printf 'MLLM_EVAL_PATH=%s\n' "$MLLM_EVAL_PATH"
printf 'MLLM_EVAL_IMAGE_DIR=%s\n' "$MLLM_EVAL_IMAGE_DIR"
```

不要打印 `HF_TOKEN`。

---

## 7. 并行启动两个实验

必须使用 `setsid`。不能只使用 `nohup ... &`。

### 实验 A：GT，GPU 0–3

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROC=4 \
BS=2 \
GA=8 \
setsid bash examples/openr1_internvl35_2b_gt.sh > gt.log 2>&1 &
```

### 实验 B：TTRL，GPU 4–7

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NUM_PROC=4 \
BS=2 \
GA=8 \
setsid bash examples/openr1_internvl35_2b_ttrl.sh > ttrl.log 2>&1 &
```

记录 shell 返回的两个后台 PID：

```bash
jobs -l
```

### 为什么必须使用 `setsid`

后台任务被回收时，系统可能连同整个进程组一起杀掉训练进程。典型症状是：

- 日志停在进度条中间；
- 没有 traceback；
- GPU 显存归零；
- 训练静默死亡。

`setsid` 可让训练脱离当前会话的进程组，降低这种风险。

---

## 8. 为什么必须使用 `BS=2 GA=8`

不要更改这两个值。

原始 8 卡配方为：

```text
bs=1 × ga=8
```

每步的 prompt 数计算公式为：

```text
prompt 数 = bs × ga × GPU 数 ÷ num_generations
```

4 卡时：

```text
2 × 8 × 4 ÷ 8 = 8 prompts
```

对应 `64 completions`，与原始配方一致。同时：

```text
num_proc × bs = 4 × 2 = 8
```

可被 `num_generations=8` 整除，满足 TRL 的硬约束。

> [!WARNING]
> 不要通过“拉大 GA”来凑等效 batch。这样会迫使 `num_generations` 降至 `2`，实际等于更换算法。GRPO 的 advantage 是在每个 prompt 的采样组内计算的，因此结果将无法与参照实验比较。详见仓库 `README §3b`。

---

## 9. 训练 budget 约束

不要传入：

```text
--max_steps
```

使用脚本中的 `num_train_epochs=1`，让训练自然完成 `722 steps`。

固定 `max_steps` 会改变训练 budget。此前某个 baseline 变成 `1.39 epoch`，正是由此导致。

---

## 10. 启动后立即验收

分别查看两个实验日志：

```bash
tail -f gt.log
```

```bash
tail -f ttrl.log
```

第 1 步即可检查以下指标。

### 10.1 Importance-sampling ratio

```text
sampling/importance_sampling_ratio/mean ≈ 0.98–1.0
```

如果该值在 `1e-5` 量级，说明以下配置未生效：

```text
--vllm_importance_sampling_mode token_truncate
```

此时应停止训练并排查配置。

### 10.2 Reward

确认以下指标均非零：

```text
rewards/reward_correctness/mean
rewards/reward_correctness/std
```

`std` 非零表示采样组内存在分歧，这是产生有效梯度的必要条件。

### 10.3 Gradient norm

确认：

```text
grad_norm != 0
```

### 10.4 Clipped ratio

检查：

```text
completions/clipped_ratio
```

第 1 步偏高（约 `30%`）属于正常现象，之后通常会下降。不要根据单步数值下结论。

---

## 11. 运行状态与 GPU 排查

### 查询实际占用 GPU 的进程

使用：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

不要只依赖：

```text
nvidia-smi --query-gpu=memory.used
```

某些 driver 会间歇性返回 `0`，造成误判。

### 检查训练进程

```bash
ps -ef | grep -E 'openr1_internvl35_2b_(gt|ttrl)|torchrun|deepspeed' | grep -v grep
```

### 需要中途重启时

重新启动任何实验之前，必须确认前一个 trainer 进程已经真正退出。否则：

- 两个 run 的日志可能交错；
- 指标会混在一起；
- 后续分析无法区分具体实验。

---

## 12. 完成条件与交付物

checkpoints 位于：

```text
work_dirs/mllm-co-grpo-dp/<run>/
```

两个实验完成后，各自收集并发送：

- `trainer_state.json`
- `train.log`

交付前确认：

- [ ] GT 实验使用 GPU 0–3，完成 `1 epoch / 722 steps`。
- [ ] TTRL 实验使用 GPU 4–7，完成 `1 epoch / 722 steps`。
- [ ] 两个实验均使用 `mmr1_8k` 预处理数据。
- [ ] 未传入 `--max_steps`。
- [ ] `sampling/importance_sampling_ratio/mean` 正常。
- [ ] reward mean/std 与 `grad_norm` 非零。
- [ ] 每个实验的 `trainer_state.json` 已收集。
- [ ] 每个实验的 `train.log` 已收集。

---

## 快速执行清单

```text
1. clone 仓库并确认 commit ≥ 3e57239
2. 按 README §3 安装完全冻结的依赖栈
3. 使用 pip constraints 锁死 torch 2.9.0+cu128
4. 预处理 MMR1，确认共 5782 条
5. 准备 MathVista testmini 的 150 张图片
6. 设置共同环境变量并确认 MLLM_PRE_DIR 指向 mmr1_8k
7. 用 setsid 在 GPU 0–3 启动 GT
8. 用 setsid 在 GPU 4–7 启动 TTRL
9. 第 1 步检查 IS ratio、reward、grad_norm 和 clipped_ratio
10. 自然跑满 1 epoch / 722 steps
11. 收集两个 run 的 trainer_state.json 和 train.log
```
