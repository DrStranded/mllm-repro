#!/usr/bin/env bash
# 自动 resume:被抢占/失败后重提 job 时,接着上次的 checkpoint 继续跑,而不是从 step 0 重来。
#
# 背景:example 脚本每次跑都用新时间戳建 run 目录,所以重提 = 从头开始。
# 自 SAVE_ONLY_MODEL 默认改为 false 后,checkpoint 里已含 optimizer/scheduler/rng
# (global_step*/ 下的 ZeRO 分片),技术上可以忠实续跑,只差"找到上次的目录并加载"。
#
# 只认**可续跑**的 checkpoint:必须同时有 trainer_state.json 和 global_step*/。
# 只存权重的(旧 SAVE_ONLY_MODEL=true 产物、或 ckpt_mirror 抄出来的扁平副本)一律跳过 ——
# 用它 resume 会拿到一个全新的优化器状态,不是原实验的忠实延续。
#
# 用法(在 example 脚本里,算出 RUN/BASE_OUT 之前):
#     source "$REPO_ROOT/../orchestration/_auto_resume.sh"   # 或绝对路径
#     ar_find_run "mmr1_internvl35_8b_ttrl_mmupt"   # 单模型
#     ar_find_run "phase4_homo_internvl35_8b_mmr1" model_a model_b   # co-learn
#     # -> 命中则设 AR_RUN_DIR(复用该目录);未命中则 AR_RUN_DIR 为空(照常新建)
#     ar_resume_arg "$BASE_OUT"          # -> AR_ARG=(--resume_from_checkpoint <ck>) 或 ()

_AR_HDFS_LIVE=/mnt/hdfs/robin/william/co-grpo/work_dirs_live/mllm-co-grpo-dp

# 某目录下最大的可续跑 checkpoint 步号(无则输出空)
_ar_max_step () {
  local d="$1" best="" s
  for c in "$d"/checkpoint-*; do
    [ -d "$c" ] || continue
    [ -f "$c/trainer_state.json" ] || continue
    compgen -G "$c/global_step*" >/dev/null 2>&1 || continue   # 没有优化器状态 = 不可续跑
    s="${c##*checkpoint-}"
    case "$s" in ''|*[!0-9]*) continue;; esac
    [ -z "$best" ] || [ "$s" -gt "$best" ] && best="$s"
  done
  echo "$best"
}

# ar_find_run <run前缀> [子目录...]   -> 设 AR_RUN_DIR / AR_RESUME_STEP
ar_find_run () {
  local prefix="$1"; shift
  local subs=("$@"); [ ${#subs[@]} -eq 0 ] && subs=("")
  AR_RUN_DIR=""; AR_RESUME_STEP=""; AR_RESUME_SRC=""
  [ "${AUTO_RESUME:-1}" = "1" ] || { echo "[resume] AUTO_RESUME=0,跳过"; return 0; }

  local roots=("${MLLM_WORKDIR_ROOT:-work_dirs}/mllm-co-grpo-dp" "$_AR_HDFS_LIVE")
  local cand step common
  # 按时间倒序遍历所有候选 run 目录,取第一个各子目录都可续跑的
  for root in "${roots[@]}"; do
    [ -d "$root" ] || continue
    # 只看**这个实验自己的固定目录**(不带时间戳)。
    # 绝不按 "${prefix}_*" 去扫历史时间戳目录 —— 那些可能是别的配置/调试残留,
    # 捡到它们会导致用错误的 LR 日程和优化器状态续跑(2026-07-26 真实踩过)。
    for cand in "$root/$prefix"; do
      [ -d "$cand" ] || continue
      common=""
      local ok=1 sd
      for sd in "${subs[@]}"; do
        step=$(_ar_max_step "${sd:+$cand/$sd}"); [ -z "$sd" ] && step=$(_ar_max_step "$cand")
        [ -n "$step" ] || { ok=0; break; }
        # co-learn 两侧必须能对齐到同一步,否则 rendezvous 会错位
        if [ -z "$common" ] || [ "$step" -lt "$common" ]; then common="$step"; fi
      done
      if [ "$ok" = 1 ] && [ -n "$common" ]; then
        # 配置一致性防呆:只有显式指定 MAX_STEPS 时才检查(生产不设,由 epochs 推导)。
        # 目的是防手工测试(如 MAX_STEPS=2)污染一个 481 步的正式 run。
        if [ -n "${MAX_STEPS:-}" ]; then
          local _ts _old
          _ts=$(ls "$cand"/checkpoint-*/trainer_state.json "$cand"/model_a/checkpoint-*/trainer_state.json 2>/dev/null | head -1)
          _old=$(sed -n 's/.*"max_steps"[: ]*\([0-9]*\).*/\1/p' "$_ts" 2>/dev/null | head -1)
          if [ -n "$_old" ] && [ "$_old" != "$MAX_STEPS" ]; then
            echo "[resume] ⚠️ $cand 的 max_steps=$_old,本次=$MAX_STEPS —— 配置不同,不复用,另起新目录"
            continue
          fi
        fi
        AR_RUN_DIR="$cand"; AR_RESUME_SRC="$cand"; AR_RESUME_STEP="$common"
        echo "[resume] 命中上次 run: $cand  (从 step $common 续跑,**写回同一目录**)"
        return 0
      fi
    done
  done
  echo "[resume] 没有可续跑的 checkpoint,从头开始"
  return 0
}

# ar_resume_arg <output_dir>  -> 设数组 AR_ARG
#
# 复用同一个 output_dir(HF 的标准续跑用法)。好处:checkpoint 编号连续、train.log 不断、
# select_best_ckpt 能看到完整历史、save_total_limit 轮转对一条序列生效、不重复占盘。
# (曾短暂改成"写新目录",依据是一次 max_steps=104 的**手工测试**写进了 481 步的目录 ——
#  那是测试污染,不是生产问题;已改回,并用上面的 MAX_STEPS 一致性检查堵住那个场景。)
ar_resume_arg () {
  local out="$1" step src
  AR_ARG=()
  [ -n "${AR_RESUME_STEP:-}" ] && [ -n "${AR_RESUME_SRC:-}" ] || return 0
  step="$AR_RESUME_STEP"
  # out 形如 <新run>/model_a;取同名子目录去旧 run 里找
  src="$out/checkpoint-$step"      # out 已是复用的旧目录
  [ -f "$src/trainer_state.json" ] || { echo "[resume] $src 不完整,该侧从头"; return 0; }
  AR_ARG=(--resume_from_checkpoint "$src")
  echo "[resume] $out <- 从 checkpoint-$step 续跑(写回同一目录)"
}

# FUSE 上 rm -rf 非空目录会 Errno 39;逐文件删再 rmdir
ar_safe_cleardir () {
  local d="$1"; [ -d "$d" ] || return 0
  find "$d" -mindepth 1 -delete 2>/dev/null || true
  rmdir "$d" 2>/dev/null || true
}
