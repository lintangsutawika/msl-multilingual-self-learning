"""MHO OpenHands-MILES training launcher.

The MILES-native launcher drives rollouts through ``harbor_agent_function.run`` (its
stock micro-swe-agent harness). This launcher instead drives rollouts through OUR
OpenHands harness by pointing Miles at ``src.harness.harbor_agent_shim.run`` — a bind-
mounted shim over ``harbor_agent_function`` that (a) registers an ``openhands-sdk``
HarnessBinding (LLM_* env + logprobs/return_token_ids + episode checker) and (b)
activates the AgentFactory swap ``openhands-sdk -> LiveRolloutOpenHandsSDK``.

Per-episode rewards come from the task verifier (``tests/test.sh`` -> reward.txt),
computed by ``src.episode.mlebench.compute_episode_reward`` (binary validate
improvement). Miles' ``apply_reward_nonzero_std_filter`` drops all-0/all-1 groups.

Qwen3.5-9B is pre-converted (§2): we ``--skip-prepare`` + ``--ref-load`` the existing
``qwen3.5-9B_torch_dist`` — never convert at runtime (the #1 failure).
"""

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer
from launch_common import agentic_pythonpath_dirs, agentic_train_args, harbor_env_vars

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_rollout_only"] = "normal"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "qwen3.5-9B"
    num_gpus_per_node: int = 8
    megatron_path: str = "/root/Megatron-LM"

    # Paths -- Qwen3.5-9B pre-converted, no runtime convert (--skip-prepare + ref-load).
    skip_prepare: bool = True
    base_dir: str = "/root"
    model_name: str = "qwen3.5-9B"
    hf_checkpoint: str = "Qwen/Qwen3.5-9B"
    ref_load: str = "/home/aci18914wh/mho-model-harness-optimization/runs_output/qwen3.5-9b_mlebench_episodes/checkpoints/qwen3.5-9B_torch_dist"
    save_dir: str = "/home/aci18914wh/mho-model-harness-optimization/runs_output/qwen3.5-9b_mlebench_episodes/checkpoints"
    prompt_data: str = (
        "/home/aci18914wh/mho-model-harness-optimization/mlebench_training_episodes/seed.jsonl"
    )

    # Training settings -- 4 tasks x 4 samples = 16 episode per async batch.
    max_seq_len: int = 32768
    num_rollout: int = 500
    rollout_batch_size: int = 4
    n_samples_per_prompt: int = 4
    global_batch_size: int = 16
    save_interval: int = 10
    save_hf: str = os.environ.get(
        "MILES_SAVE_HF",
        "/home/aci18914wh/mho-model-harness-optimization/runs_output/qwen3.5-9b_mlebench_episodes/checkpoints/hf/step_{}",
    )

    # Harbor settings (see harbor_agent_shim.py + run_training.py for semantics).
    harbor_env_type: str = os.environ.get("HARBOR_ENV_TYPE", "singularity")
    harbor_env_kwargs: str = os.environ.get("HARBOR_ENV_KWARGS", "")
    harbor_tasks_dir: str = os.environ.get(
        "HARBOR_TASKS_DIR",
        "/home/aci18914wh/mho-model-harness-optimization/mlebench_training_episodes",
    )
    harbor_trials_dir: str = os.environ.get("HARBOR_TRIALS_DIR", "/tmp/harbor_trials")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "qwen3.5-9B")
    agent_timeout: int = int(os.environ.get("AGENT_TIMEOUT", "5400"))

    modal_config_file: str = os.environ.get("MODAL_CONFIG_PATH", "")
    wandb_key: str = os.environ.get("WANDB_KEY", os.environ.get("WANDB_API_KEY", ""))
    wandb_project: str = os.environ.get("WANDB_PROJECT", "mho-harness")
    wandb_run_name: str = "qwen3.5-9b-mlebench-openhands"


def cleanup():
    """Kill old Ray jobs / stale sglang+Megatron processes to free GPUs."""
    my_pid = os.getpid()
    ppid = os.getppid()
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in ["sglang", "train.py", "MegatronTrain"]:
        subprocess.run(f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true", shell=True)
    time.sleep(5)


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--save {args.save_dir} "
        f"--save-interval {args.save_interval} "
        + (f"--save-hf {args.save_hf} " if args.save_hf else "")
    )
    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--apply-chat-template "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        "--rollout-max-response-len 8192 "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )
    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )
    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-tool-call-parser qwen3_coder "
        "--sglang-reasoning-parser qwen3 "
        "--sglang-router-port 31000 "
    )
    # Agent wiring: our OpenHands shim over Miles' harbor_agent_function.
    agent_args = agentic_train_args(tito_model="qwen35", session_server_workers=32)
    agent_args = agent_args.replace(
        "--custom-agent-function-path harbor_agent_function.run",
        "--custom-agent-function-path src.harness.harbor_agent_shim.run",
    )
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
        "--rollout-num-gpus 6 "
    )
    debug_args = "--debug-rollout-only " if args.mode == "debug_rollout_only" else ""
    # In debug-rollout-only mode we do not need the 2 train GPUs; free them so a
    # 1-2 GPU smoke can validate the rollout/OpenHands path without a full 8-GPU alloc.
    if args.mode == "debug_rollout_only":
        misc_args = misc_args.replace("--actor-num-gpus-per-node 2 ", "")
    wandb_args = ""
    if args.wandb_key:
        wandb_args = f"--use-wandb --wandb-project {args.wandb_project} --wandb-group {args.wandb_run_name} --wandb-key {args.wandb_key} "

    train_args = (
        f"{ckpt_args}{rollout_args}{optimizer_args}{grpo_args}{wandb_args}"
        f"{perf_args}{sglang_args}{agent_args}{misc_args}{debug_args}"
    )
    # The rollout worker must import our OpenHands harness: src/harness/harbor_agent_shim
    # (registers the openhands-sdk binding), src/harness/backend (AgentFactory swap), and
    # src/episode/mlebench (episode boundary + reward). Prepend the MSL repo src/ to the
    # MILES PYTHONPATH that agentic_pythonpath_dirs() yields.
    _msl_src = str(Path(__file__).resolve().parent)  # <repo>/src
    extra_env_vars = {
        "PYTHONPATH": ":".join([_msl_src, args.megatron_path, *agentic_pythonpath_dirs(), str(U.repo_base_dir)]),
        **harbor_env_vars(args),
        # Our OpenHands harness knobs on the rollout worker:
        "AGENT_MODEL_NAME": args.agent_model_name,
        "LITELLM_EXTRA_BODY": '{"return_token_ids": true, "logprobs": true}',
        "EPISODE_CHECKER_MODULE": "src.episode.mlebench",
    }
    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
        train_script="train_async.py",
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup()
    # Always skip runtime HF->Megatron conversion; we ref-load the pre-converted
    # qwen3.5-9B_torch_dist (see module docstring + miles_training.md §2).
    args.skip_prepare = True
    execute(args)


if __name__ == "__main__":
    main()