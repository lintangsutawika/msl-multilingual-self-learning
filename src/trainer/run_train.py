"""MILES trainer launcher for the lifelong loop: PPO on episodes the rollout job records in the spool."""
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import typer
from launch_common import agentic_pythonpath_dirs, harbor_env_vars
import miles.utils.external_utils.command_utils as U

@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: str = "train"
    run_id: str = U.create_run_id()
    megatron_model_type: str = "qwen3.5-9B"
    num_gpus_per_node: int = int(os.environ.get("NUM_GPUS", "8"))
    megatron_path: str = "/root/Megatron-LM"
    skip_prepare: bool = True
    base_dir: str = "/root"
    model_name: str = "qwen3.5-9B"
    # Keep the Hub id: --critic-load reuses this, and a local HF dir there makes MILES take its
    # HF loader (bridge mode only). HF exports get config/tokenizer copied by the publisher instead.
    hf_checkpoint: str = os.environ.get("HF_CHECKPOINT", "Qwen/Qwen3.5-9B")
    _msl_repo: str = os.environ.get("MSL_REPO", str(Path(__file__).resolve().parents[1]))
    # Base model inputs on group storage (MSL_BASE): the Megatron torch_dist conversion of the
    # base HF model (actor init + KL reference) and the placeholder prompt file MILES requires
    # (the spool rollout fn ignores its contents).
    _msl_base: str = os.environ.get("MSL_BASE", "/groups/gai51740/aci18914wh/msl/base")
    ref_load: str = os.environ.get("REF_LOAD", f"{_msl_base}/qwen3.5-9B_torch_dist")
    save_dir: str = os.environ.get("SAVE", f"{_msl_repo}/runs/qwen3.5-9b_mlebench_episodes/checkpoints")
    episodes_dir: str = os.environ.get("EPISODES_DIR", f"{_msl_repo}/runs/qwen3.5-9b_mlebench_episodes/episodes")
    max_seq_len: int = 32768
    global_batch_size: int = int(os.environ.get("GLOBAL_BATCH_SIZE", "4"))
    num_rollout: int = int(os.environ.get("NUM_ROLLOUT", "16"))
    save_interval: int = int(os.environ.get("SAVE_INTERVAL", "1"))
    save_hf: str = os.environ.get("MILES_SAVE_HF", f"{_msl_repo}/runs/qwen3.5-9b_mlebench_episodes/checkpoints/hf/step_{{rollout_id}}")  # MILES formats with save_hf.format(rollout_id=...)
    harbor_env_type: str = os.environ.get("HARBOR_ENV_TYPE", "sail")
    harbor_env_kwargs: str = os.environ.get("HARBOR_ENV_KWARGS", "")
    harbor_tasks_dir: str = os.environ.get("HARBOR_TASKS_DIR", f"{_msl_repo}/harbor_tasks/mlebench-low")
    harbor_trials_dir: str = os.environ.get("HARBOR_TRIALS_DIR", f"{_msl_repo}/runs/qwen3.5-9b_mlebench_episodes/harbor_trials")
    prompt_data: str = os.environ.get("PROMPT_DATA", f"{_msl_base}/seed_list.jsonl")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "qwen3.5-9B")
    agent_timeout: int = int(os.environ.get("AGENT_TIMEOUT", "5400"))
    modal_config_file: str = os.environ.get("MODAL_CONFIG_PATH", "")
    router_external_host: str = os.environ.get("MILES_ROUTER_EXTERNAL_HOST", "127.0.0.1")
    miles_host_ip: str = os.environ.get("MILES_HOST_IP", "")
    wandb_key: str = os.environ.get("WANDB_KEY", "")
    wandb_project: str = os.environ.get("WANDB_PROJECT", "msl-lifelong")
    wandb_run_name: str = os.environ.get("WANDB_RUN_NAME", "trainer")

def cleanup():
    for t in ["sglang", "train.py", "MegatronTrain"]:
        subprocess.run("pgrep -f '%s' | xargs -r kill 2>/dev/null || true" % t, shell=True)
    time.sleep(3)

def execute(args: ScriptArgs):
    ckpt_args = (f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ref_load} --save {args.save_dir} --save-interval {args.save_interval} " + (f"--save-hf {args.save_hf} " if args.save_hf else ""))
    # Long context (episodes reach 150k+ tokens; Qwen3.5-9B allows 262144): context parallelism
    # splits each sequence across CP_SIZE GPUs (MILES passes GDN state between CP ranks);
    # dynamic batching packs samples up to MAX_TOKENS_PER_GPU per micro-batch. Logprobs are
    # computed in LOG_PROBS_CHUNK_SIZE-token chunks: full-vocab logits for an 80k-token
    # micro-batch are ~16 GB and ran out of memory unchunked.
    # MILES places the critic on the SAME GPUs as the actor (it expects --offload-train to
    # page one out while the other trains; we keep offload off, see below), so both models
    # are resident on all TRAIN_GPUS: shard each across all of them.
    tp, cp = os.environ.get("TP_SIZE", "2"), os.environ.get("CP_SIZE", "1")
    dyn = "--use-dynamic-batch-size " if os.environ.get("DYNAMIC_BATCH", "1") == "1" else ""
    perf_args = ("--tensor-model-parallel-size %s --sequence-parallel --pipeline-model-parallel-size 1 --context-parallel-size %s --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 --micro-batch-size 1 --max-tokens-per-gpu %s --log-probs-chunk-size %s %s--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer " % (tp, cp, os.environ.get("MAX_TOKENS_PER_GPU", "16384"), os.environ.get("LOG_PROBS_CHUNK_SIZE", "4096"), dyn))
    # MILES turns --offload-train on by default (model memory paged out between steps); the trainer
    # runs no SGLang, so it buys nothing here, and with it the HF export read paged-out params
    # and wrote all-zero weights (2026-11-03). Only an explicit --no-offload-train disables it.
    # Opt back in with OFFLOAD_TRAIN=1.
    offload = "--offload-train " if os.environ.get("OFFLOAD_TRAIN") == "1" else "--no-offload-train "
    ppo_args = ("--advantage-estimator ppo --critic-load %s %s--use-rollout-logprobs --use-kl-loss --kl-loss-coef 0.01 --kl-loss-type low_var_kl --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 " % (args.hf_checkpoint, offload))
    optimizer_args = ("--optimizer adam --lr %s " % os.environ.get("LR", "1e-6") + "--lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 ")
    if os.environ.get("ONLINE_MODE"):
        # Online: the trainer never generates. The rollout fn reads closed episodes that the rollout job's
        # forwarder captured into the shared spool. Critic (--advantage-estimator ppo) computes
        # values in-loop; --use-rollout-logprobs makes the captured behavior logprobs the ratio denominator.
        fn = os.environ.get("ROLLOUT_FUNCTION_PATH", "src.trainer.spool_rollout.generate_rollout")
        ingest_args = (
            "--debug-train-only "  # the trainer runs no sglang; rollouts come from the rollout job via the rollout fn
            # After each save: copy config/tokenizer into the HF export, publish LATEST for the
            # rollout job's weight watcher, prune old checkpoints (src/trainer/publisher.py).
            "--custom-megatron-post-save-hook-path src.trainer.publisher.post_save "
            "--rollout-function-path %s "
            "--prompt-data %s --input-key prompt --metadata-key metadata "
            "--rollout-shuffle "
            "--global-batch-size %s --rollout-batch-size %s --num-rollout %s " % (fn, args.prompt_data, os.environ.get("ROLLOUT_BATCH_SIZE", "8"), os.environ.get("ROLLOUT_BATCH_SIZE", "8"), os.environ.get("NUM_ROLLOUT", "16"))
        )
        print("[run_train] ONLINE_MODE: rollout-function-path=%s spool=%s" % (fn, os.environ.get("SPOOL_DIR", "")), flush=True)
    else:
        ingest_args = ("--load-debug-rollout-data %s/{rollout_id}.pt --prompt-data %s --input-key prompt --metadata-key metadata --rollout-shuffle --global-batch-size %s --rollout-batch-size %s --num-rollout %s " % (args.episodes_dir, args.prompt_data, args.global_batch_size, args.global_batch_size, args.num_rollout))
    misc_args = ("--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash --actor-num-nodes 1 --actor-num-gpus-per-node %s " % os.environ.get("TRAIN_GPUS", "8"))
    # W&B: on unless WANDB_MODE=disabled. Without an explicit key, wandb uses the ~/.netrc login.
    wandb_args = ""
    if os.environ.get("WANDB_MODE") != "disabled":
        wandb_args = "--use-wandb --wandb-project %s --wandb-group %s " % (args.wandb_project, args.wandb_run_name)
        key = args.wandb_key or os.environ.get("WANDB_API_KEY", "")
        if key:
            wandb_args += "--wandb-key %s " % key
    train_args = ckpt_args + perf_args + ppo_args + optimizer_args + ingest_args + misc_args + wandb_args
    _msl_src = str(Path(__file__).resolve().parent)
    extra_env_vars = {"PYTHONPATH": ":".join([_msl_src, args.megatron_path, *agentic_pythonpath_dirs(), str(U.repo_base_dir)]), **harbor_env_vars(args), "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"), "EPISODE_CHECKER_MODULE": "src.episode.mlebench", **{k: v for k, v in os.environ.items() if k.startswith("SPOOL_") or k in ("METRIC_DIRECTIONS", "HF_BASE_DIR", "KEEP_HF", "KEEP_CKPT")}}
    U.execute_train(train_args=train_args, config=args, num_gpus_per_node=args.num_gpus_per_node, megatron_model_type=args.megatron_model_type, megatron_path=args.megatron_path, extra_env_vars=extra_env_vars, train_script="train_async.py")

@U.dataclass_cli
def main(args: ScriptArgs):
    cleanup(); args.skip_prepare = True; execute(args)

if __name__ == "__main__":
    main()
