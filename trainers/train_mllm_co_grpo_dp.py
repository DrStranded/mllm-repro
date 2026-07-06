"""Entry point for one half of a cross-supervised multimodal GRPO run.

Launched twice in parallel (once per group) by a `dp-scripts/` launcher.
Each launch is an independent accelerate world bound to its own
CUDA_VISIBLE_DEVICES and master port. The two launches coordinate solely
through the file rendezvous directory (`--rendezvous_dir`) to exchange
pseudo-labels every generation step.

Differences from `co-grpo-dp/train_co_grpo_dp.py`:
  1. `AutoProcessor` (VLM) instead of `AutoTokenizer` (LLM).
  2. `train_dataset` choices: CLEVR-Counting / GEOQA (not math).
  3. `extract_boxed_answer` here extracts from `<answer>...</answer>`
     (R1-V convention) instead of `\\boxed{}` (math convention). Same
     function name to mirror co-grpo-dp's reward-function structure.
  4. `grade_answer` here is backed by `math_verify` (R1-V baseline
     grader) instead of qwen-sympy.

All other args (group / rendezvous / wandb / training_args / dual-seed
trick for vLLM divergence / etc.) are identical to co-grpo-dp.
"""

import os
import json
import shutil
from dataclasses import dataclass, field

import wandb
import torch.nn as _nn
from co_label_utils import extract_boxed_answer
from mcq_grade import grade_mcq_or_math
from dataset import CLEVR_COUNTING_DATASET, GEOQA_DATASET, load_dataset
from mllm_co_grpo_dp_trainer import CoGRPOdpTrainer
from transformers import AutoProcessor
from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from rendezvous import Rendezvous


# Gemma-3 + ZeRO-3 fix: PreTrainedModel._init_weights for nn.Embedding does
# `module.weight.data[module.padding_idx].zero_()`. Under ZeRO-3, non-rank-0
# processes see size-0 weight shards because deepspeed.zero.GatheredParameters
# only materializes on modifier_rank=0. Indexing into a size-0 tensor crashes
# with `IndexError: index 0 is out of bounds for dimension 0 with size 0`.
# Qwen2.5-VL embedding has padding_idx=None so its base init never hits this
# branch; Gemma-3 sets padding_idx and crashes.
_orig_init_weights = _PreTrainedModel._init_weights


def _safe_init_weights(self, module):
    if isinstance(module, _nn.Embedding) and module.weight.data.numel() == 0:
        return
    return _orig_init_weights(self, module)


_PreTrainedModel._init_weights = _safe_init_weights


from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class MllmCoGRPOdpScriptArguments(ScriptArguments):
    """Script arguments for mllm-co-grpo-dp (single-model, one-group-per-launch)."""

    group: str = field(
        default=None,
        metadata={"help": "'A' or 'B' — which half of the cross-supervision run this launch is."},
    )
    rendezvous_dir: str = field(
        default=None,
        metadata={"help": "Directory shared between groups for pseudo-label exchange."},
    )
    peer_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Peer group's model id (for logging only; peer is launched separately)."},
    )
    run_config: str = field(
        default=None,
        metadata={"help": "Run name prefix for this experiment."},
    )
    wandb_entity: str = field(default=None, metadata={"help": "WandB entity."})
    wandb_project: str = field(default="mllm-co-grpo-dp", metadata={"help": "WandB project name."})
    train_dataset: str = field(
        default=CLEVR_COUNTING_DATASET,
        metadata={
            # No `choices` whitelist: `dataset.load_dataset` is the single source
            # of truth for supported sources (GeoQA/CLEVR + the `_SPECS` datasets)
            # and raises a clear ValueError for anything unknown. (Matches
            # train_mllm_single.py; the whitelist here blocked zwz/openr1/etc.)
            "help": "Dataset to use for training (see dataset._SPECS for sources).",
        },
    )
    self_consistency_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Minimum top-answer frequency for a pseudo-label to be accepted. "
            "0.0 accepts the plurality winner; 0.5 requires a strict majority."
        },
    )
    log_oracle_accuracy: bool = field(
        default=True,
        metadata={"help": "Log how often pseudo-labels match real ground truth (diagnostic only)."},
    )


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts.
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def reward_correctness(completions, solution, **kwargs):
    """Reward function: 1.0 if completion's parsed answer is math-equivalent
    to the (peer-supplied or ground-truth) solution, else 0.0.

    `solution` here can be:
      - train mode: peer's pseudo-label (from majority vote), possibly the
        sentinel `_UNLABELED_SENTINEL` for prompts the peer dropped — sentinel
        cannot match any parseable answer, so reward is 0 for those.
      - eval mode: dataset's real ground-truth solution (eval branch in
        trainer skips the cross-labeling override).

    Uses `math_verify.verify` (HuggingFace official, same as R1-V baseline)
    so equivalent forms like `1/2` vs `\\frac{1}{2}` vs `0.5` all count as
    correct. Slower than string equality (~1-10ms per check) but eliminates
    spurious negative rewards on format-only diffs.
    """
    rewards = []
    for completion, ground_truth in zip(completions, solution):
        pred_answer = extract_boxed_answer(_get_text(completion))
        if pred_answer is not None and grade_mcq_or_math(pred_answer, ground_truth):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


class BestKeeperCallback(TrainerCallback):
    """DeepSpeed-compatible substitute for `load_best_model_at_end=True`,
    which HF Trainer rejects when combined with DeepSpeed/FSDP +
    `save_only_model=True`. On every save, if the latest eval metric beat the
    prior best, hardlink the just-written checkpoint to `$output_dir/best_model/`
    (0 byte / 0 time via inode refcount; survives ring-buffer deletion of the
    source ckpt). Copied verbatim from train_mllm_single.py (= text repo's).
    """

    def __init__(self, metric_name="eval_reward", greater_is_better=True):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best = None
        self.last_metrics = {}

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics:
            self.last_metrics = metrics

    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return
        v = self.last_metrics.get(self.metric_name)
        if v is None:
            # TRL merges reward metrics into the logged dict (state.log_history),
            # not into the metrics handed to on_evaluate; fall back to the most
            # recent logged value so best-ckpt tracking actually fires.
            for entry in reversed(state.log_history):
                if self.metric_name in entry:
                    v = entry[self.metric_name]
                    break
        if v is None:
            return
        better = self.best is None or (
            (v > self.best) if self.greater_is_better else (v < self.best)
        )
        if not better:
            return
        self.best = v
        src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        dst = os.path.join(args.output_dir, "best_model")
        if not os.path.exists(src):
            return
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, copy_function=os.link)
        with open(os.path.join(args.output_dir, "best_metric.json"), "w") as f:
            json.dump(
                {"step": state.global_step, "metric": self.metric_name, "value": float(v)},
                f, indent=2,
            )


if __name__ == "__main__":
    parser = TrlParser((MllmCoGRPOdpScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if script_args.group not in ("A", "B"):
        raise ValueError(f"--group must be 'A' or 'B', got {script_args.group!r}")

    # Group B uses an offset `seed` so the two groups' vLLM/torch RNG diverge.
    # Without this, both groups' accelerate worlds set torch.manual_seed(seed +
    # process_index) with identical (seed, process_index) pairs, producing byte-
    # identical vLLM rollouts and forcing peer_agreement → 1 (cross-supervision
    # degenerates into self-vote).
    # IMPORTANT: do NOT also bump `data_seed`. `data_seed` is the
    # transformers-convention sampler seed; both groups must iterate the
    # dataset in identical order so that `gathered_answers[g*G:(g+1)*G]`
    # corresponds to the SAME prompt on A and B (required for cross-
    # supervision to be meaningful). See trl/trainer/grpo_trainer.py
    # `_get_train_sampler` — it reads `data_seed` when set, otherwise falls
    # back to `seed`. If `data_seed` is None here, bumping `seed` alone
    # would also misalign prompts; set it explicitly.
    if script_args.group == "B":
        if training_args.data_seed is None:
            training_args.data_seed = training_args.seed
        training_args.seed += 1
    if script_args.rendezvous_dir is None:
        raise ValueError("--rendezvous_dir is required for mllm-co-grpo-dp.")

    ################
    # WandB
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    model_short = model_args.model_name_or_path.split("/")[-1]
    peer_short = (
        script_args.peer_model_name_or_path.split("/")[-1]
        if script_args.peer_model_name_or_path
        else "unknown"
    )

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_group{script_args.group}_lr{lr_str}_bs{effective_batch_size}"
    else:
        full_wandb_run_name = (
            f"MllmCoGRPOdp_{model_short}_x_{peer_short}_group{script_args.group}_"
            f"lr{lr_str}_bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}_"
            f"sct{script_args.self_consistency_threshold}"
        )

    print(f"\n{'='*80}")
    print(f"MLLM-CO-GRPO-DP (group {script_args.group}) CONFIGURATION")
    print(f"{'='*80}")
    print(f"This model   : {model_args.model_name_or_path}")
    print(f"Peer model   : {script_args.peer_model_name_or_path}")
    print(f"Dataset      : {script_args.train_dataset}")
    print(f"Rendezvous   : {script_args.rendezvous_dir}")
    print(f"WandB run    : {full_wandb_run_name}")
    print(f"Output dir   : {training_args.output_dir}")
    print(f"SCT          : {script_args.self_consistency_threshold}")
    print(f"World size   : {num_processes}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "group": script_args.group,
                "model": model_args.model_name_or_path,
                "peer_model": script_args.peer_model_name_or_path,
                "train_dataset": script_args.train_dataset,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "loss_type": training_args.loss_type,
                "scale_rewards": training_args.scale_rewards,
                "steps_per_generation": training_args.steps_per_generation,
                "vllm_importance_sampling_correction": training_args.vllm_importance_sampling_correction,
                "adam_beta2": training_args.adam_beta2,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "lr_scheduler_kwargs": training_args.lr_scheduler_kwargs,
                "warmup_ratio": training_args.warmup_ratio,
                "max_grad_norm": training_args.max_grad_norm,
                "weight_decay": training_args.weight_decay,
                "eval_steps": training_args.eval_steps,
                "num_generations_eval": training_args.num_generations_eval,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "data_seed": training_args.data_seed,
                "self_consistency_threshold": script_args.self_consistency_threshold,
                "vllm_gpu_memory_utilization": training_args.vllm_gpu_memory_utilization,
                "seed": training_args.seed,
            },
        )

    ################
    # Model & Processor (VLM)
    ################
    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        # `use_cache` deliberately omitted: InternVL3.5-HF's
        # `InternVLForConditionalGeneration.__init__` rejects a `use_cache` kwarg
        # (TypeError at create_model_from_path). GRPO never uses the cache during
        # training anyway. Kept in lockstep with train_mllm_single.py.
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # `AutoProcessor` rather than `AutoTokenizer` — handles both the
    # tokenizer (for text) and the image processor (for vision tower
    # input) in a single API. `processing_class` on the trainer accepts
    # either tokenizer or processor; for VLMs GRPOTrainer routes through
    # `processor.tokenizer` for text ops and `processor.image_processor`
    # for image preprocessing.
    #
    # Plain AutoProcessor — kept in lockstep with train_mllm_single.py. The old
    # `load_processor_for_mllm` wrapper force-loaded InternVL's *custom-code*
    # `modeling_internvl_chat.py` (a transformers-5.x post_init patch), which
    # the HF-native `InternVL3_5-*-HF` repos do not ship → OSError at step 0.
    # HF-native InternVL needs only the crop_to_patches fix below, same as single.
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    # AutoProcessor exposes the underlying tokenizer at `processor.tokenizer`.
    # GRPO left-pads completions and expects pad_token to be set.
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # InternVL3.5-HF crashes step 0 with "Image features and image tokens do not
    # match: tokens: 3328, features 256" because InternVLProcessor defaults
    # crop_to_patches=True → 1 image → up to 13 tiles in pixel_values; TRL's
    # split_pixel_values_by_grid only handles Qwen image_grid_thw / Gemma
    # image_position_ids, returns the batch unchanged, then split_tensor_dict
    # naively chunks pixel_values by shape[0]/num_chunks and drops most tiles.
    # Fix: force no-tiling on processor instance + class-level kwargs defaults.
    # (Verbatim from train_mllm_single.py — consistency mandatory.)
    if "internvl" in model_args.model_name_or_path.lower():
        if hasattr(processor, "image_processor"):
            if hasattr(processor.image_processor, "crop_to_patches"):
                processor.image_processor.crop_to_patches = False
            if hasattr(processor.image_processor, "max_patches"):
                processor.image_processor.max_patches = 1
            if hasattr(processor.image_processor, "min_patches"):
                processor.image_processor.min_patches = 1
        try:
            from transformers.models.internvl.processing_internvl import InternVLProcessorKwargs
            InternVLProcessorKwargs._defaults["images_kwargs"]["crop_to_patches"] = False
        except Exception:
            pass

    # Gemma3-IT uses <end_of_turn> (id=106) as the turn terminator, but HF
    # tokenizer.eos_token_id still returns 1 (<eos>). Patch both tokenizer and
    # generation_kwargs so TRL and vLLM agree on the stop token set.
    # (Verbatim from train_mllm_single.py — consistency mandatory.)
    _model_name_lower = model_args.model_name_or_path.lower()
    if "gemma-3" in _model_name_lower or "gemma3" in _model_name_lower:
        _GEMMA3_EOT_ID = 106  # <end_of_turn>
        processor.tokenizer.eos_token_id = _GEMMA3_EOT_ID
        processor.tokenizer.eos_token = "<end_of_turn>"
        _existing = training_args.generation_kwargs or {}
        training_args.generation_kwargs = {**_existing, "stop_token_ids": [1, _GEMMA3_EOT_ID]}

    ################
    # Dataset — two groups use the same seed/world_size so RepeatSampler
    # yields identical index sequences, ensuring both groups train on the
    # same prompts at every generation step (required for cross-labeling).
    ################
    train_dataset, eval_dataset = load_dataset(script_args.train_dataset)

    ################
    # PEFT
    ################
    peft_config = get_peft_config(model_args)

    ################
    # Rendezvous
    ################
    rendezvous = Rendezvous(
        rendezvous_dir=script_args.rendezvous_dir,
        my_group_name=script_args.group,
    )

    ################
    # Training
    ################
    trainer = CoGRPOdpTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=peft_config,
        my_group_name=script_args.group,
        rendezvous=rendezvous,
        self_consistency_threshold=script_args.self_consistency_threshold,
        log_oracle_accuracy=script_args.log_oracle_accuracy,
    )

    trainer.add_callback(BestKeeperCallback())

    trainer.train()
    trainer.save_model(training_args.output_dir)
