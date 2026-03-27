# Reproduced Performance

Reproduction of published VLA model benchmark scores using vla-eval.

## Measurement Protocol

**Hardware:**
- Model server: H100-80GB SXM GPU
- Benchmark host: 96-core Xeon, 2× A100-80GB PCIe, 503GB RAM

**Software:**
- Harness: vla-eval `main` branch
- Docker: `ghcr.io/allenai/vla-evaluation-harness/{benchmark}:latest` (rebuilt per evaluation)

**Verdict criteria** (binomial 95% CI):
- **Reproduced**: within 95% CI of reported score.
- **Approximate**: outside CI but ≤5pp gap.
- **Not reproduced**: >5pp gap, or known systematic issue.

---

## Stage 1 — LIBERO

**Protocol:** 4 suites (Spatial, Object, Goal, 10) × 10 tasks × 50 episodes = 2000 episodes/model.
Seed=7, num_steps_wait=10, max_steps per suite (Spatial=220, Object=280, Goal=300, 10=520).

### Results

Models listed in publication order.

| Model | Spatial | Object | Goal | 10 | **Avg** | Reported | Verdict |
|-------|:-------:|:------:|:----:|:--:|:-------:|:--------:|:-------:|
| Pi0.5 (Oct 2024) | 98.0% | 99.6% | 98.6% | 94.6% | **97.7%** | 96.9% | Reproduced |
| OFT (Feb 2025, joint) | 94.0% | — | — | — | **—** | ~96.8% | Spatial only (−3.6pp) |
| GR00T N1.6 (Mar 2025) | 96.6% | 98.4% | 96.8% | 87.8% | **94.9%** | 97.0% | Approximate (−2.1pp) |
| X-VLA (Oct 2025) | 98.0% | 98.0% | 98.0% | 94.8% | **97.2%** | 98.1% | Reproduced |

Each value = successful episodes / total episodes (500 per suite).
Raw result JSONs: [`data/`](data/).

### Per-model Reproduction Notes

**Pi0.5** (`pi05_libero` via openpi, [arxiv 2410.24164](https://arxiv.org/abs/2410.24164)):
- Uses `states` (from `raw_obs`), 8D `[pos3, axisangle3, gripper2]`.
- `send_wrist_image=True`, `send_state=True`. `image_resolution=224`.
- Note: `pi0_fast_libero` is a different, lower-performing model.

**OFT** (`moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10`, [arxiv 2502.19645](https://arxiv.org/abs/2502.19645)):
- Joint checkpoint, requires per-suite `unnorm_key` — 4 server instances or 4 sequential runs.
- `num_images_in_input=2` (3rd-person + wrist), `send_state=True`.
- TF CUDA JIT on H100 takes 30+ min at startup. Ensure server is ready before launching shards.
- Per-suite checkpoints (`moojink/openvla-7b-oft-finetuned-libero-{suite}`) show anomalous results
  (spatial 92%, goal 18%) — likely HuggingFace checkpoint issues. Use joint checkpoint instead.

**GR00T N1.6** (`0xAnkitSingh/GR00T-N1.6-LIBERO`, [arxiv 2503.14734](https://arxiv.org/abs/2503.14734)):
- Community checkpoint; `invert_gripper=True`: model outputs gripper [0,1] (0=close), LIBERO expects [-1,1] (-1=open).
- `embodiment_tag=LIBERO_PANDA`, `chunk_size=16`.
- −2.1pp gap vs reported may be due to community checkpoint vs official NVIDIA finetuning.

**X-VLA** (`2toINF/X-VLA-Libero`, [arxiv 2510.10274](https://arxiv.org/abs/2510.10274)):
- `benchmark_profile=libero`. Uses `controller_states` (from `robot.controller.ee_pos/ee_ori_mat`),
  NOT `states` (from `raw_obs`). The observation quaternion (`robot0_eef_quat`) differs from the
  controller rotation matrix by ~90° due to coordinate frame differences. X-VLA was trained on
  controller data; using observation data yields 42%. See commit `27e63c0`.
- `unflip_wrist=True`: benchmark flips all images; X-VLA was trained with unflipped wrist.
- `absolute_action=True`: X-VLA outputs absolute EE poses, not deltas.
- All params auto-negotiated via HELLO (`get_observation_params()`).

### Excluded Models

| Model | Reason |
|-------|--------|
| StarVLA Q2.5-OFT/Q3-OFT/Q2.5-FAST | Supply <7 obs/s (chunk_size=1). Single shard: 4-20 hours. |
| StarVLA Q2.5-GR00T | Supply 38 obs/s but still slow. Partial result: 29.6% spatial (reported 95.4%). |
| StarVLA Qwen3-PI | state_dict mismatch: 36 vs 16 DiT transformer blocks. Server crash. |
| OpenVLA base (LoRA) | chunk_size=1, no batch prediction. ~3 hours per suite. |

### Bugs Found During Reproduction

| Bug | Impact | Fix |
|-----|--------|-----|
| `raw_obs["robot0_eef_quat"]` ≠ `robot.controller.ee_ori_mat` (~90° frame diff) | X-VLA 42% → 98% | Benchmark sends both; X-VLA reads `controller_states` |
| StarVLA gripper polarity `2x-1` inverted | Gripper open/close swapped | Changed to `1-2x` |
| GR00T missing gripper normalization/inversion | ~1% success | Added `invert_gripper` flag |
| OFT `num_images_in_input=1` (should be 2) | Missing wrist image | Fixed in `_base.yaml` |
| Pi0 default `pi0_fast_libero` (wrong model) | Wrong checkpoint loaded | Changed to `pi05_libero` |
| Smoke test `success` check broken after metrics refactor | All smoke tests failing | Fixed to read `metrics.success` |
| Shard merge dropped `server_info`, `harness_version`, `created_at` | Provenance lost | Fixed in `merge.py` |
| Port conflicts: multiple servers on same port | OFT evaluated against wrong models | Run models sequentially, verify ports |

---

## Stage 2 — Cross-benchmark

### Overview

Models listed in publication order.

| Model | LIBERO | CALVIN | SE WidowX | SE GR-VM | RoboTwin |
|-------|:------:|:------:|:---------:|:--------:|:--------:|
| GR00T N1.6 (Mar 2025) | 94.9% | — | — | — | — |
| X-VLA (Oct 2025) | 97.2% | — | — | — | — |
| DB-CogACT (Oct 2025) | 95.2% | 4.05 avg len | 72.2% | — | — |

LIBERO column from Stage 1. DB-CogACT details: [db-cogact.md](db-cogact.md).

### CALVIN (ABC→D)

**Protocol:** 1000 sequences × 5 chained subtasks, max 360 steps/subtask.
Docker: `ghcr.io/allenai/vla-evaluation-harness/calvin:latest`.
Config: `configs/calvin_eval.yaml`.
Demand: Peak λ=407 obs/s at N=24 (CPU-bottlenecked).

| Model | Checkpoint | Config | 1/5 | 2/5 | 3/5 | 4/5 | 5/5 | **Avg Len** | Reported | Verdict |
|-------|-----------|--------|:---:|:---:|:---:|:---:|:---:|:-----------:|:--------:|:-------:|
| X-VLA | `2toINF/X-VLA-Calvin-ABC_D` | `xvla/calvin.yaml` | — | — | — | — | — | **—** | 4.43 | — |
| DB-CogACT | `Dexmal/calvin-db-cogact` | `db_cogact/calvin.yaml` | 93.3% | 86.3% | 81.5% | 75.6% | 68.4% | **4.05** | 4.06 | Reproduced |

**X-VLA notes:**
- Checkpoint: `2toINF/X-VLA-Calvin-ABC_D`, `benchmark_profile=calvin`, `domain_id=2`.
- `chunk_size=20`, flow matching with `denoising_steps=10`.
- Uses predicted proprio (closed-loop), `send_wrist_image=True`, `send_state=True`.
- Config ready: `configs/model_servers/xvla/calvin.yaml`.

### SimplerEnv — WidowX VM

**Protocol:** 4 tasks × 24 episodes × 3 seeds (0, 2, 4) = 288 episodes/model.
Docker: `ghcr.io/allenai/vla-evaluation-harness/simpler:latest`.
Config: `configs/simpler_all_tasks.yaml`.
Demand: Peak λ=138 obs/s at N=24 (GPU-bottlenecked).

| Model | Checkpoint | Config | Spoon | Carrot | Block | Eggplant | **Avg** | Reported | Verdict |
|-------|-----------|--------|:-----:|:------:|:-----:|:--------:|:-------:|:--------:|:-------:|
| GR00T N1.6 | `nvidia/GR00T-N1.6-bridge` | config needed | — | — | — | — | **—** | 62.1%† | — |
| X-VLA | `2toINF/X-VLA-WidowX` | config needed | — | — | — | — | **—** | 95.8% | — |
| DB-CogACT | `Dexmal/simpler-db-cogact` | `db_cogact/simpler.yaml` | 94.4% | 72.2% | 25.0% | 97.2% | **72.2%** | 69.5% | Reproduced |

† GR00T uses non-standard 7-task set (includes open/close drawer). 4-task subset avg = 57.1%.

**GR00T N1.6 notes:**
- Checkpoint: `nvidia/GR00T-N1.6-bridge`, `embodiment_tag=OXE_WIDOWX`.
- Config needed: create `configs/model_servers/groot/simpler.yaml`.
- Non-standard task set in official eval; we evaluate the standard 4-task subset.

**X-VLA notes:**
- Checkpoint: `2toINF/X-VLA-WidowX`, `benchmark_profile=simpler`.
- Config needed: create `configs/model_servers/xvla/simpler.yaml` with `domain_id` TBD.

### SimplerEnv — Google Robot VM

**Protocol:** Standard 3-task (Pick Coke Can, Move Near, Open/Close Drawer) × 24 episodes × 3 seeds.
Docker: `ghcr.io/allenai/vla-evaluation-harness/simpler:latest`.

| Model | Checkpoint | Pick Coke | Move Near | Drawer | **3-task Avg** | Reported | Verdict |
|-------|-----------|:---------:|:---------:|:------:|:--------------:|:--------:|:-------:|
| GR00T N1.6 | `nvidia/GR00T-N1.6-fractal` | — | — | — | **—** | 67.7%† | — |
| X-VLA | `2toINF/X-VLA-WidowX` | — | — | — | **—** | 88.3% (VM) | — |

† GR00T uses non-standard 6-task set. Not directly comparable.

**GR00T N1.6 notes:**
- Checkpoint: `nvidia/GR00T-N1.6-fractal`, `embodiment_tag=OXE_GOOGLE`.

### RoboTwin

**Protocol:** 50 tasks, Protocol A (single-task, 50 clean demos/task).
Docker: `ghcr.io/allenai/vla-evaluation-harness/robotwin:latest`.
Config: `configs/robotwin_eval.yaml`.
Demand: Peak λ=4.9 obs/s at N=16 (GPU-bottlenecked). Rec. 2 GPUs.

| Model | Checkpoint | Easy | Hard | Reported Easy | Reported Hard | Verdict |
|-------|-----------|:----:|:----:|:-------------:|:-------------:|:-------:|
| X-VLA | `2toINF/X-VLA-WidowX` | — | — | 70.0% | 39.0% | — |

**X-VLA notes:**
- `benchmark_profile=robotwin`, `domain_id` TBD.
- Config needed: create `configs/model_servers/xvla/robotwin.yaml`.
- X-VLA model server already supports `robotwin` benchmark profile.
- RoboTwin demand is very low (4.9 obs/s peak) — few shards needed.

---

## Supply — Model Server Throughput

Measured with `experiments/bench_supply.py` on H100-80GB SXM.
Command: `uv run python experiments/bench_supply.py --url ws://HOST:PORT --num-clients 4 --requests-per-client 60 --image-size 256`
Observation payload: 2× 256×256 RGB images (agentview + wrist) + 8D state.
All models at `max_batch_size=1` (no batching).

| Model | chunk_size | μ (obs/s) | Median latency |
|-------|:---------:|:---------:|:--------------:|
| X-VLA | 30 | 88.8 | 30ms |
| Pi0.5 | 10 | 84.0 | 63ms |
| GR00T N1.6 | 16 | 46.5 | 50ms |
| StarVLA Q2.5-GR00T | 1 | 38.3 | 60ms |
| OFT (joint) | 10 | 27.1 | 46ms |
| StarVLA Q2.5-OFT | 1 | 6.0 | 654ms |
| StarVLA Q3-OFT | 1 | 5.9 | 664ms |
| StarVLA Q2.5-FAST | 1 | 1.4 | 2858ms |

StarVLA/GR00T support `predict_batch()`. X-VLA/Pi0/OFT are single-predict only.

## Demand — Benchmark Observation Rate

Measured with `experiments/bench_demand.py` on the benchmark host.
Command: `uv run python experiments/bench_demand.py --config CONFIG --shards N --episodes-per-shard 5 --gpus G --timeout 300`
Median CPU/GPU utilization during steady-state (startup transients excluded).

Full per-N sweep data: see [`../tuning-guide.md`](../tuning-guide.md).

### Bottleneck Summary

| Benchmark | Peak λ (obs/s) | Peak N | Bottleneck | 2 GPU effect | Rec. GPUs |
|-----------|:--------------:|:------:|:----------:|:------------:|:---------:|
| LIBERO | 415 | 50 | CPU (52%) | No change | 1 |
| CALVIN | 407 | 24 | CPU (93%) | No change | 1 |
| SimplerEnv | 138 | 24 | GPU (43%) | Worse (overhead) | 1 |
| RoboTwin | 4.9 | 16 | GPU (100%) | 2× improvement | 2 |

## How to Run

```bash
# 1. Build Docker
docker/build.sh libero

# 2. Start model server (slurm, one per GPU)
sbatch --gres=gpu:1 -c8 --mem=64G -t 24:00:00 \
  --wrap="uv run vla-eval serve -c configs/model_servers/xvla/libero.yaml --address 0.0.0.0:8001 -v"

# 3. Wait for server ready
curl -s --max-time 2 "http://GPU-NODE:8001/config"

# 4. Run benchmark (ONE MODEL AT A TIME — shard filenames collide across models)
SHARDS=10  NODE=GPU-NODE  MODEL=xvla
for i in $(seq 0 $((SHARDS-1))); do
  uv run vla-eval run -c configs/libero_all.yaml \
    --server-url ws://${NODE}:8001 \
    --shard-id $i --num-shards $SHARDS --yes &
done
wait

# 5. Archive shards + merge
mkdir -p docs/reproductions/data/${MODEL}-libero/shards
cp results/LIBEROBenchmark_*shard*of${SHARDS}.json docs/reproductions/data/${MODEL}-libero/shards/
uv run vla-eval merge results/LIBEROBenchmark_*_shard*of${SHARDS}.json \
  -o docs/reproductions/data/${MODEL}-libero/merged.json
rm results/LIBEROBenchmark_*shard*of${SHARDS}.json

# 6. Next model — clean shards before starting
```

**Critical notes:**
- Run ONE model at a time. Merge and clean shards before the next.
- Max 50 Docker containers on benchmark host.
- Verify server port is free before launching (no stale servers on same port).
- OFT: TF JIT takes 30+ min. Confirm server ready via `curl /config` before launching shards.
- OFT joint: requires per-suite unnorm_key. Run 4 sequential passes, or 4 server instances on different ports.

## Reference

- [reported-performance.md](reported-performance.md) — Officially reported scores from papers/model cards.
- [db-cogact.md](db-cogact.md) — DB-CogACT cross-benchmark reproduction report.
- [`../tuning-guide.md`](../tuning-guide.md) — Supply/demand measurement methodology.
