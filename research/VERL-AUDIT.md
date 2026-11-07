# verl 0.9.0 audit against the GRPO specification

Purpose: state, with code citations, where verl 0.9.0 already matches `project.md` Sections 6 and 8, where it differs, and what we must override, add or test. The target setup is Qwen3-1.7B, full-parameter GRPO, one 48 GB GPU, with vLLM rollout colocated on that GPU.

**Source.** Tag `v0.9.0` of github.com/volcengine/verl, commit `483b8a009ba3a97563edee3a19887e4862b8094a` (2026-08-14). `verl/version/version` reads `0.9.0`. I read the code and did not run it. All paths are relative to the repository root. Line numbers refer to that commit. Audit date: 2026-11-06.

**Which trainer runs.** `trainer.use_v1` defaults to `true` (`verl/trainer/config/ppo_trainer.yaml:222`). `verl/trainer/main_ppo.py:184-193` then runs `TaskRunnerV1` with `trainer.v1.trainer_mode=sync` (`ppo_trainer.yaml:228`, class `PPOTrainerSync` in `verl/trainer/ppo/v1/trainer_sync.py:24-42`). The legacy `RayPPOTrainer` (`verl/trainer/ppo/ray_trainer.py:285-286`) is marked deprecated. This audit covers the V1 sync path. Both paths use the same engine workers (`verl/workers/engine_workers.py`) and loss code (`verl/workers/utils/losses.py`, `verl/trainer/ppo/core_algos.py`).

**Terms.** A *round* is one rollout of 16 prompts × 8 responses followed by the optimizer updates on those 128 responses. In verl, one round is one `global_step`. A *minibatch* is the set of responses for one optimizer step. A *micro-batch* is one forward/backward pass inside a minibatch.

---

## 1. GRPO advantage

**Answer.** The built-in `grpo` estimator uses the **Bessel-corrected** standard deviation (`unbiased=True`), not the population standard deviation. Epsilon is `1e-6`. An all-equal group gets advantage exactly 0. A custom estimator can be registered by name, so we can get the exact specification without patching verl.

| Item | Evidence |
|---|---|
| `grpo`: `id2std[idx] = torch.std(scores_tensor)`. `torch.std` defaults to `unbiased=True`. | `verl/trainer/ppo/core_algos.py:321` |
| `grpo`: `(scores[i] - mean) / (std + epsilon)`, `epsilon: float = 1e-6` | `core_algos.py:272`, `:326` |
| `grpo_vectorized` is also Bessel-corrected: `var = var_num / (count - 1).clamp_min(1)` | `verl/utils/groupwise.py:209-212`; called at `core_algos.py:352` |
| Singleton group: mean 0, std 1 | `core_algos.py:315-317` |
| Scalar is broadcast over valid response tokens: `scores.unsqueeze(-1) * response_mask` | `core_algos.py:329` |
| Score = sum of `token_level_rewards` over the response. The reward is placed on the last response token. | `core_algos.py:304`; `verl/experimental/agent_loop/agent_loop.py:142-147` |

**All-equal groups.** For binary rewards, every `r - mean` is exactly 0.0, so A = 0/(0 + 1e-6) = 0 exactly. The behaviour matches the specification. The difference is in mixed groups. For n = 8, Bessel std = population std × √(8/7). Every non-zero advantage is therefore 0.935× the specified value. That works like a 6.5% smaller learning rate on every non-degenerate group.

**Registry API.**
- Decorator `register_adv_est(name)` and lookup `get_adv_estimator_fn(name)` (`core_algos.py:113-150`). Registering the same name twice with a different function raises `ValueError` (`:127-130`).
- Select it with `algorithm.adv_estimator=<name>` (`ppo_trainer.yaml:74`).
- In V1, `_compute_advantage` (`verl/trainer/ppo/v1/trainer_base.py:1620-1629`) sends any name other than `grpo` to `compute_advantage` (`verl/trainer/ppo/v1/utils.py:167-176`). That function then calls the custom function with keyword arguments `token_level_rewards`, `response_mask`, `config` (the `algorithm` config) and `index` (the `uid` array) (`verl/trainer/ppo/ray_trainer.py:248-281`). It does **not** pass `epsilon` or `norm_adv_by_std_in_grpo`. The function must return `(advantages, returns)`, each shaped `(bs, response_len)`.
- The registry is a module-level dict, so the module must be imported **inside the Ray `TaskRunnerV1` process**. Importing it in the driver is not enough. Two hooks import it there:
  - `actor_rollout_ref.model.external_lib=<module>`. `HFModelConfig.__post_init__` calls `import_external_libs` (`verl/workers/config/model.py:149`). The trainer builds this config in the TaskRunner (`trainer_base.py:648`) before it computes any advantage. Recommended.
  - Environment variable `VERL_USE_EXTERNAL_MODULES=<module>`, read on `import verl` (`verl/__init__.py:38-41`). The variable must reach the Ray actors, for example through `ray_kwargs.ray_init.runtime_env.env_vars`.

**Consequence for our implementation.**
1. Register `grpo_popstd` with `A = (r - mean) / (std(unbiased=False) + 1e-6)`, grouped by `index`. Set `algorithm.adv_estimator=grpo_popstd` and import it through `model.external_lib`.
2. Do not use `grpo`, `grpo_vectorized` or `norm_adv_by_std_in_grpo=false`. The last option is Dr.GRPO: it uses no std at all.
3. Set `critic.enable=false` explicitly. For a non-GAE estimator, `need_critic` turns the critic off but warns (`verl/trainer/ppo/utils.py:96-107`).
4. Test: a group of 8 with 3 correct answers must give A = ±values computed with population std, within 1e-6. An all-0 group and an all-1 group must give exactly 0. Also check that the run actually used the registered function, for example by logging its name.

---

## 2. Loss aggregation and accumulation

**Answer.** `actor.loss_agg_mode=seq-mean-token-mean` gives "mean over valid tokens per response, then mean over responses". The default is `token-mean` (`verl/trainer/config/actor/actor.yaml:86`). That default is a global token mean over the minibatch and does not match the specification.

**How `seq-mean-token-mean` is computed** (`core_algos.py:1194-1202`):
```
seq_losses = sum(loss_mat * mask, -1) / (sum(mask, -1) + 1e-8)     # per-response token mean
loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
```
`global_batch_size` and `dp_size` come from `config.global_batch_info` (`verl/workers/utils/losses.py:64-68`). The policy loss passes them to `agg_loss` (`core_algos.py:1367-1369`).

**What sets `global_batch_size`.**
- The trainer sets `global_batch_size = mini_batch_size = actor.ppo_mini_batch_size × rollout.n` (`trainer_base.py:1674-1675`, `:1696-1697`). With our settings that is 4 × 8 = 32 responses.
- `dp_size` is the data-parallel size, which is 1 on one GPU (`verl/workers/engine/fsdp/transformer_impl.py:711`).

**Accumulation.**
- `prepare_micro_batches` cuts the minibatch into `len / micro_batch_size_per_gpu` equal chunks when `use_dynamic_bsz=false` (`verl/workers/engine/utils.py:122-129`).
- Each micro-batch's loss is back-propagated as returned, with no extra scaling (`transformer_impl.py:725-738`).
- The optimizer steps once per minibatch: `optimizer_zero_grad` → `forward_backward_batch` → `optimizer_step` (`verl/workers/engine/base.py:126-128`).
- With a micro-batch of 1 response, each micro-batch therefore contributes `seq_mean_i / 32`. The 32 backward passes add up to exactly the mean over the 32 responses. A short final micro-batch cannot get a different weight, because the denominator is the fixed 32, not the micro-batch count.

**Global-token normalisation.** `forward_backward_batch` always computes `batch_num_tokens` over the whole minibatch (`transformer_impl.py:705-710`). Only `token-mean` uses it (`core_algos.py:1170-1175`). `seq-mean-token-mean` ignores it, so it does not break our reduction.

**Zero-variance groups.** A response with A = 0 adds 0 to the numerator and still counts in the fixed denominator of 32. This matches the specification. Only fully masked responses are excluded from `seq_mask`, and the fixed `global_batch_size` overrides that count anyway (`:1197-1201`).

**Side effects to know.**
- With a micro-batch of 1, policy-loss metrics such as `actor/pg_clipfrac` are averaged per micro-batch (`AggregationType.MEAN`, `losses.py:116`). They become a mean over responses, not a token-weighted mean.
- A non-finite gradient norm silently skips the update after printing `WARN` (`transformer_impl.py:793-798`).

**Consequence.**
1. Set `actor.loss_agg_mode=seq-mean-token-mean`, `actor.ppo_micro_batch_size_per_gpu=1`, `actor.use_dynamic_bsz=false`.
2. Test on a fixture: the accumulated gradient over 32 micro-batches of 1 must equal the gradient of one 32-response pass with the specified loss. Compare in fp32 with a tolerance.
3. Test that micro-batch sizes 1, 2 and 4 give the same gradient.
4. Log non-finite skips as an anomaly. verl does not count them.

---

## 3. Minibatch semantics, shuffling, ordering

**Answer.** `actor.ppo_mini_batch_size` counts **prompts**. The trainer multiplies it by `rollout.n` (`trainer_base.py:1674-1675`). `data.train_batch_size` also counts prompts. `ppo_micro_batch_size_per_gpu` counts **responses**: it is passed straight to the engine (`engine_workers.py:621-622`) and used to chunk responses.

**Settings for 4 optimizer steps of 32 responses each, one pass.**

| Key | Value | Where read |
|---|---|---|
| `data.train_batch_size` | 16 | `trainer_base.py:510`, `:1330` |
| `actor_rollout_ref.rollout.n` | 8 | `verl/trainer/ppo/v1/agent_loop_tq.py:115` |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 4 (→ 32 responses) | `trainer_base.py:1674-1675` |
| `actor_rollout_ref.actor.ppo_epochs` | 1 | `trainer_base.py:1698` |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 1 | `engine_workers.py:621-622` |
| `trainer.v1.sync` | leave empty (`parameter_sync_step` = 1) | `trainer_base.py:134`, `:509-534` |

`train_mini_batch` then iterates 128 / 32 = 4 minibatches (`engine_workers.py:262-288`).

**LR scheduler.** The scheduler steps once per round, not once per optimizer step (`update_lr_scheduler` is true only on the last minibatch, `engine_workers.py:304-308`, `:367-370`). With a constant LR this has no effect.

**Shuffling.**
- `actor.shuffle` defaults to `false` (`actor.yaml:122`).
- `data_loader_seed: 42` (`actor.yaml:125`) seeds a `torch.Generator`. That generator is used only when shuffle is on (`verl/utils/tensordict_utils.py:593-605`).
- With `shuffle=false`, minibatch k is rows `[32k, 32k+32)` of the batch, in the batch's current row order (`tensordict_utils.py:601-610`).

**Batch row order is not under our control by default.**
- Each response is written to the TransferQueue (TQ), the key-value store that V1 uses to pass data between rollout and training, **as soon as it finishes** (`agent_loop_tq.py:150-227`).
- The replay buffer builds the batch by iterating the TQ listing (`verl/trainer/ppo/v1/replay_buffer.py:377-389`). That listing comes from the external package `TransferQueue==0.1.8`. Its iteration order is **unknown**; it may be completion order.
- Groups can therefore be interleaved in the batch. The 4 minibatches would then not be complete groups, and the 2-hard + 2-auxiliary composition would not hold.
- verl has no configuration key that fixes this.

**`balance_batch`.**
- In V1, `_balance_batch` runs **unconditionally**: it ignores `trainer.balance_batch` (`trainer_base.py:556`, `:1453-1477`).
- With dp_size = 1, Karmarkar–Karp yields one partition, and `_check_and_sort_partitions` sorts its indices (`verl/utils/seqlen_balancing.py:241-254`). The reorder is therefore the identity on one GPU.
- It also pads the batch up to a multiple of lcm(dp, 32) (`trainer_base.py:1434-1451`; `verl/trainer/ppo/padding_utils.py:127-198`). With 128 responses no padding is added.
- If a group is missing responses, synthetic padding rows are appended at the end and the minibatch boundaries shift.

**Failed groups.** In sync mode a group in which any session raised is marked `failure`. By default it stays sampleable with its missing responses, or it is refilled with a *different* prompt (`replay_buffer.py:93-113`, `:448-450`; `agent_loop_tq.py:131-148`). Either case breaks "complete groups" and the specification's "never silently drop and replace".

**Consequence.**
1. Write a custom `ReplayBuffer` subclass and set `trainer.v1.sampler.custom_sampler.path` and `.name` (`ppo_trainer.yaml:263-272`; loaded at `trainer_base.py:148-165`). Its `sample()` must:
   - wait for all 16 groups;
   - assert 16 groups × 8 sessions, all with status `finished`, and otherwise raise to hold the round;
   - return keys ordered so that each block of 32 is [hard group, hard group, aux group, aux group], with each group's 8 sessions contiguous and in a seeded, recorded order.

   The class can read dataset fields (`index`, `extra_info`, `data_source`) from the TQ values, because each trajectory value carries the prompt's fields (`agent_loop_tq.py:194-195`). The constructor receives `trainer_mode`, `trainer_config`, `max_off_policy_threshold`, `max_off_policy_strategy`, `sampler_kwargs`, `refill_fn` (`trainer_base.py:158-165`).
2. Keep `actor.shuffle=false`.
3. Test on the real pipeline: log the uid and role of every row in each minibatch for one round and assert the composition. Unit fixtures are not enough, because the dispatch path from TQ keys to worker rows is not audited here.

---

## 4. Old log probabilities and the ratio denominator

**Answer.** By default ("decoupled" mode, `algorithm.rollout_correction.bypass_mode=false`, `verl/trainer/config/algorithm/rollout_correction.yaml`), `old_log_probs` comes from an **FSDP actor forward pass**. That pass runs once per round, before the first update, on the same weights that vLLM sampled with (`trainer_base.py:558-560`, `:1495-1516`). The four minibatches use this tensor unchanged. The ratio in the loss is `exp(log_prob - old_log_prob)` (`core_algos.py:1336-1339`). verl's π_old is therefore the actor's teacher-forced log probability, not vLLM's.

**vLLM log probabilities.**
- `rollout.calculate_log_probs: True` is the default (`verl/trainer/config/rollout/rollout.yaml:233`). The agent loop requests `logprobs` (`agent_loop_tq.py:64-70`) and stores them as `rollout_log_probs` (`agent_loop.py:124-126`).
- `rollout.logprobs_mode: processed_logprobs` (`rollout.yaml:93`). At temperature 1.0 and top_p 1.0, these equal the raw log probabilities.

**Built-in comparison.**
- `calculate_debug_metrics` logs `training/rollout_probs_diff_{max,mean,std}` and `training/rollout_actor_probs_pearson_corr` (`trainer_base.py:1534-1536`; `verl/utils/debug/metrics.py:63-113`). These are differences of **probabilities**, not log probabilities, and give no quantiles.
- The default `rollout_correction` block also runs on every round. With `rollout_is: null` and `rollout_rs: null` it changes neither the mask nor the weights. It logs `rollout_corr/kl`, `k3_kl`, `log_ppl_diff`, `chi2_token` and related metrics (`trainer_base.py:1610-1617`; `verl/trainer/ppo/rollout_corr_helper.py:836-901`).

**Options.**
- `bypass_mode=true` sets `old_log_probs = rollout_log_probs`, so q = π_θ / π_vLLM (`trainer_base.py:1485-1493`).
- `rollout_is: token|sequence` with `rollout_is_threshold` gives truncated importance sampling (TIS) weights that multiply the per-token loss (`core_algos.py:1363-1365`).
- `rollout_rs` gives rejection masks.

**Consequence.**
1. Decide which quantity is "π_rollout" in the specification:
   - Default decoupled mode uses the actor recompute (same weights, different kernels).
   - `bypass_mode=true` uses the stored vLLM log probabilities literally.

   Both are fixed across the 4 steps. Record the choice in `DECISIONS.md`.
2. Keep `rollout_is=null` and `rollout_rs=null`. Otherwise the objective changes.
3. Add our own per-round mismatch record: quantiles of |log π_actor − log π_vLLM| per valid token, computed before the first update. This is the Section 8 check, and verl does not provide it.

---

## 5. Clipping

**Names and defaults.**
- `actor.clip_ratio` 0.2, `clip_ratio_low` 0.2, `clip_ratio_high` 0.2 (`actor.yaml:36-42`).
- `actor.clip_ratio_c` 3.0 (`actor.yaml:83`).
- `actor.policy_loss.loss_mode: vanilla`.

**Dual clip is always on for A < 0.** `compute_policy_loss_vanilla` computes `clip_pg_losses2 = min(-A·c, clip_pg_losses1)` and selects it wherever `advantages < 0` (`core_algos.py:1355-1361`). An assertion requires `clip_ratio_c > 1.0` (`:1331-1334`), so dual clip cannot be switched off by configuration. It can only be pushed out of reach: with `clip_ratio_c=1e10` it never activates, because the ratio is capped at exp(20) ≈ 4.85e8.

**A second deviation.** The log-ratio is clamped to [−20, 20] before `exp` (`core_algos.py:1336-1338`). Beyond that range the gradient is zero. For A < 0 and q > e^20 the specified objective still has a gradient. This is an edge case, but it is a deviation.

**Metrics.**
- `actor/pg_clipfrac`: fraction of valid tokens where the clipped branch is strictly larger, `gt(pg_losses2, pg_losses1)`. It mixes both advantage signs (`:1353`).
- `actor/pg_clipfrac_lower`: despite its name, this is the **dual-clip activation** fraction (A < 0 and q > c), **not** the 1−ε_low clip fraction (`:1357-1359`).
- `actor/ppo_kl`: mean of −log q (`:1340`).
- verl gives no split by advantage sign and no lower-bound or upper-bound fractions.

**Consequence.** Register our own policy loss with `register_policy_loss("spec_clip")` (`core_algos.py:53-67`) and set `actor.policy_loss.loss_mode=spec_clip`. The module must also be imported in the actor worker; `model.external_lib` does this, see `engine_workers.py:534`. The loss must:
1. compute exactly `-min(q·A, clip(q, 1−ε_low, 1+ε_high)·A)`, with no dual clip, and decide explicitly whether to keep the ±20 clamp;
2. aggregate with `agg_loss(..., **config.global_batch_info)` so that Section 2 still holds;
3. return clip fractions split by sign: A>0 with q>1+ε_high, and A<0 with q<1−ε_low, plus the totals.

Test it against a hand-computed fixture that covers each branch.

---

## 6. KL, entropy and the reference model

| Spec | verl key, default | Evidence |
|---|---|---|
| No KL loss | `actor.use_kl_loss: false` | `actor.yaml:103`; used only if true, `losses.py:132-142` |
| No KL in reward | `algorithm.use_kl_in_reward: False` | `ppo_trainer.yaml:98`; `trainer_base.py:1599-1605` |
| No entropy bonus | `actor.entropy_coeff: 0`, `calculate_entropy: false` | `actor.yaml:93-96`; `losses.py:123-129` |

**Reference model.** It is **not instantiated** when both KL flags are false:
- `need_reference_policy` returns false (`verl/trainer/ppo/utils.py:75-79`);
- the role becomes `ActorRollout`, not `ActorRolloutRef` (`trainer_base.py:746`);
- the ref model is built only when `"ref" in self.role` (`engine_workers.py:537`).

**Entropy is still computed during the old-log-prob pass.** That pass computes entropy for the `actor/entropy` metric (`trainer_base.py:1496-1532`). This costs compute but does not change the loss.

**Consequence.** Set all three values explicitly. Add a startup assertion that no ref worker exists and that the loss has no `kl_loss` or `entropy_loss` term.

---

## 7. Dropout

**Answer.**
- Training runs with `module.train()` (`transformer_impl.py:1105`). The old-log-prob pass runs with `module.eval()` (`:1079`). verl does not override dropout for language models; it zeroes dropout only for value heads (`:277-280`).
- Qwen3 has no dropout layers. Attention dropout comes from `config.attention_dropout`. In the published `Qwen/Qwen3-1.7B/config.json` that value is `0.0`. I fetched the config from huggingface.co on 2026-11-06 and did not pin a revision.
- Effective dropout is therefore 0 in both passes, but only by the checkpoint's configuration.
- vLLM rollout has no dropout.

**Consequence.**
1. Set `actor_rollout_ref.model.override_config.attention_dropout=0.0` explicitly. It is merged into the HF config at `verl/workers/config/model.py:216-220`.
2. Test on the pinned revision: after model construction, `attention_dropout == 0.0`, and no `nn.Dropout` module has p > 0.
3. Test that a train-mode forward equals an eval-mode forward on the same tokens.

---

## 8. Optimizer, precision, offload

**Optimizer.**
- `actor.optim.optimizer: AdamW`, `optimizer_impl: torch.optim`.
- `betas: [0.9, 0.999]`, `weight_decay: 0.01` (**must change to 0**).
- `lr_scheduler_type: constant`, `lr_warmup_steps: -1`, `lr_warmup_steps_ratio: 0.0`.
- `clip_grad: 1.0`.
- Sources: `verl/trainer/config/optim/fsdp.yaml:5-39`; `verl/workers/config/optimizer.py:47-61`, `:106-114`.
- `build_optimizer` passes only `lr`, `weight_decay` and `betas`. `eps` takes the `torch.optim.AdamW` default of 1e-8 unless set through `optim.override_optimizer_config` (`optimizer.py:326-336`).
- Constant schedule without warmup: warmup steps ≤ 0 makes verl use `ratio × total` = 0, then `get_constant_schedule_with_warmup` (`transformer_impl.py:484-503`).

**Gradient clipping.** The engine reads `actor.optim.clip_grad` (`transformer_impl.py:766-783`). The key `actor.grad_clip` (`verl/trainer/config/actor/dp_actor.yaml:29`) exists, but grep finds no reader in the V1 FSDP path. Treat it as unused.

**Precision.**
- `actor.fsdp_config.model_dtype: fp32` (`verl/trainer/config/engine/fsdp.yaml:33`). The model is loaded in fp32 (`transformer_impl.py:239-245`, `:314`). These fp32 parameters are the master weights.
- FSDP `MixedPrecision(param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=fp32)` is the default when `mixed_precision` is null (`transformer_impl.py:372-382`).
- The forward runs under `torch.autocast(bf16)` (`:1517-1523`).
- AdamW holds fp32 states because the parameters are fp32.
- On one GPU, FSDP1 uses `FULL_STATE_DICT` (`:462-467`).
- This matches "BF16 computation, FP32 optimizer states/master weights".

**Offload.**
- `actor.fsdp_config.param_offload` and `optimizer_offload` (`fsdp.yaml:12-16`) are verl-managed CPU moves between phases.
- `offload_policy` (`fsdp.yaml:19`) applies to FSDP2 only.
- FSDP1 CPUOffload is forced off for the actor because of gradient-accumulation errors (`transformer_impl.py:403-416`).

**Other defaults.** Gradient checkpointing (`hf_model.yaml:34`) and `use_remove_padding` (`hf_model.yaml:40`) default to true. Attention uses `flash_attention_2` unless `override_config.attn_implementation` is set (`model.py:185`).

**Consequence.**
1. Set `optim.weight_decay=0`, `optim.override_optimizer_config={eps: 1e-8}`, `optim.lr_warmup_steps=0`, `optim.lr_scheduler_type=constant`, `optim.clip_grad=1.0`.
2. Test that the built optimizer's `param_groups` show these values and fp32 state tensors.
3. Test that every parameter has `requires_grad`.
4. Record the attention kernel actually used.

---

## 9. Single-GPU colocated memory path

**Round sequence** (`trainer_sync.py:31-42`, `trainer_base.py:509-586`):
1. After any checkpoint load, actor weights are pushed to vLLM (`on_init_end`).
2. Each round: prompts are submitted, then all 16 groups are awaited.
3. `on_sample_end` → `sleep_replicas`. In hybrid mode this calls `engine.sleep(level=2)`, which drops vLLM weights and KV cache (`verl/workers/rollout/vllm_rollout/vllm_async_server.py:1075-1098`).
4. Old-log-prob pass and actor update.
5. `on_step_end` → `update_weights`: resume vLLM weights, copy actor → vLLM per tensor, resume KV cache (`engine_workers.py:720-805`).

Weights are therefore synchronised once per round, after the four optimizer steps, as the specification requires. vLLM starts with `load_format: dummy` (`rollout.yaml:100`). Correctness depends on the first sync in `on_init_end`.

**Keys and defaults that matter on one GPU.**

| Key | Default | Needed |
|---|---|---|
| `trainer.n_gpus_per_node` | 8 | 1 |
| `rollout.tensor_model_parallel_size` | **2** (`rollout.yaml:64`) | 1 |
| `rollout.gpu_memory_utilization` | 0.5 (`rollout.yaml:46`) | measured value |
| `rollout.free_cache_engine` | True (`rollout.yaml:61`) | True |
| `rollout.enable_sleep_mode` | True (`verl/workers/config/rollout.py:268`) | True |
| `rollout.max_model_len` | null → `max_position_embeddings` = 40960 (`vllm_async_server.py:954-955`) | 5120 (1024 + 4096), or 9216 for an 8k cap |
| `rollout.max_num_seqs` / `max_num_batched_tokens` | 256 / 8192 (`rollout.yaml:78-84`) | lower first on OOM |
| `rollout.log_prob_micro_batch_size_per_gpu` | null; **required** when `use_dynamic_bsz=false` (`engine_workers.py:629-631`) | set |
| `rollout.name` | `???` (must be set) | `vllm` |

**Memory estimate (unmeasured).** Qwen3-1.7B has about 1.7 × 10⁹ parameters.
- fp32 parameters: about 6.9 GB.
- AdamW states: about 13.8 GB.
- fp32 gradients during the update: about 6.9 GB.
- Without offload, about 21 GB of parameters and optimizer state stay resident while vLLM generates.
- vLLM at 0.5 × 48 GB = 24 GB then leaves only a few GB of headroom, plus a CUDA context per process.

**Consequence.**
1. Measure peak memory for generation and for the update with 128 responses at the 4,096-token cap. verl logs `perf/max_memory_allocated_gb` (`engine_workers.py:218-220`).
2. If memory does not fit, follow the specification's order: lower `max_num_seqs` and `max_num_batched_tokens` first, then enable `param_offload` and `optimizer_offload`.
3. Test that one update changes vLLM's output log probabilities on a fixed prompt after the sync (Section 8).

---

## 10. Checkpointing and resume

**Saved** (`trainer_base.py:885-957`). Saving is controlled by `trainer.save_freq`, which defaults to −1 (off), and happens after the round's update.
- Actor model, optimizer and "extra" state (`actor.checkpoint.save_contents: [model, optimizer, extra]`, `actor.yaml:135`). Extra holds the LR scheduler state and the actor process's RNG states (CPU torch, CUDA, NumPy, Python `random`) (`verl/utils/checkpoint/fsdp_checkpoint_manager.py:355-361`; `checkpoint_manager.py:197-207`).
- `data.pt`: the state of the `StatefulDataLoader`, including its sampler position (`trainer_base.py:933-936`).
- `latest_checkpointed_iteration.txt`, skipped when `async_save` is on (`:948-957`).

**Resume** (`trainer_base.py:789-845`).
- `trainer.resume_mode`: `auto` (latest in `default_local_dir`), `disable`, or `resume_path`; default `auto` (`ppo_trainer.yaml:179-186`).
- `global_steps` is parsed from the folder name. The actor checkpoint and the dataloader state are reloaded.
- In sync mode no TQ state is saved, so a partly finished round is discarded and regenerated.

**Not restored.**
- The TaskRunner process RNG. Group uids are `uuid4` and not seeded (`trainer_base.py:1325`).
- vLLM sampling RNG. vLLM restarts with engine seed `replica_rank + rollout.seed` (`vllm_async_server.py:306`), so generations after a resume differ from an uninterrupted run.
- The W&B run identity. `wandb.init` is called without `id` or `resume` (`verl/utils/tracking.py:84`); continuity needs the `WANDB_RUN_ID` and `WANDB_RESUME` environment variables.
- Per-request generation seeds. They exist only with `rollout.full_determinism=true` (`vllm_async_server.py:578-580`), and then every request gets the **same** seed. With n = 8 identical prompts, this may produce 8 identical samples. Also, `request_id` becomes `det-{priority}` (`verl/experimental/agent_loop/single_turn_agent_loop.py:66`). **Do not enable it without a test.**

**Consequence.**
1. Set `trainer.save_freq=1`, or another frequency we choose.
2. Store our own sidecar state with each checkpoint: schedule position, master-seed streams, counters, W&B run id.
3. Run the Section 8 forced-interruption test. Assert no skipped or duplicated round by comparing the logged prompt ids per round.
4. Record the vLLM RNG limitation in the run notes.

---

## 11. Custom sampler and dataset hooks

**Answer.**
- There is **no `data.sampler.class_path`** in 0.9.0. Grep over `verl/` and `docs/` finds no `class_path`.
- The prompt sampler is fixed in `create_rl_sampler`: a `torchdata` `RandomSampler` when `data.shuffle=True` (the default, `verl/trainer/config/data/legacy_data.yaml:68`), else `SequentialSampler` (`verl/trainer/ppo/utils.py:140-168`; used at `trainer_base.py:694`).
- `data.seed` defaults to `null`, which gives an **unseeded** generator (`legacy_data.yaml:71`; `utils.py:159-162`).

**Available hooks.**
1. `data.custom_cls.path` / `.name`: a custom `torch.utils.data.Dataset` (`legacy_data.yaml:107-113`; `verl/utils/dataset/rl_dataset.py:566-585`).
2. `trainer.v1.sampler.custom_sampler`: a custom `ReplayBuffer` that orders responses *after* generation (Section 3).

Each round takes `train_batch_size` consecutive items from the dataloader (`trainer_base.py:1315-1343`). The iterator restarts at `StopIteration`. `drop_last=True` (`:688-695`).

**Consequence.**
1. Materialise the full per-round prompt schedule (8 hard + 8 auxiliary per round, seeded and recorded) as an ordered dataset.
2. Set `data.shuffle=false` and `data.train_batch_size=16`.
3. Set `trainer.total_training_steps` explicitly. `steps_per_epoch = len(dataset) // 16` (`:710-716`).
4. Test that the prompts of round k equal schedule rows [16k, 16k+16), including after a resume.

---

## 12. Reward hook and truncation

**Custom function.**
- `reward.custom_reward_function.path` / `.name`, plus optional `reward_kwargs` (`verl/trainer/ppo/reward.py:50-86`).
- The default naive manager calls `compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)`. It may be sync or async, and returns a float or a dict containing `"score"` (`verl/experimental/reward_loop/reward_manager/naive.py:66-97`).
- `ground_truth` is `reward_model["ground_truth"]` from the dataset row (`:43`).
- `extra_info` is the row's `extra_info` plus `num_turns`, `rollout_reward_scores` and the tool extra fields (`:44-52`).
- verl adds **no timeout** around the call.

**Truncation.**
- **Not identifiable inside `compute_score`.** `solution_str` is decoded with `skip_special_tokens=True` (`naive.py:54-56`), so the EOS token is gone.
- The response length is not passed.
- vLLM's `finish_reason` collapses `"stop"` and `"length"` into `stop_reason="completed"` (`vllm_async_server.py:657-664`). The single-turn loop does not forward `stop_reason` anyway (`single_turn_agent_loop.py:91-110`).

**What a custom reward manager can see.** A manager loaded with `reward.reward_manager.source=importlib`, `.module.path`, `.name` (`reward.py:89-108`) receives the response token ids and the attention mask (`naive.py:37-40`; built in `agent_loop.py:969-1013`). It can define truncated = (no EOS token at the end) and (length == `max_response_length`). EOS for Qwen3 is ids 151645 and 151643 (`generation_config.json`). The single-turn loop also cuts responses to `response_length` (`single_turn_agent_loop.py:97`).

**Related trap: the prompt cap.**
- The V1 agent loop does not check the prompt length.
- `max_tokens = min(response_length, prompt_length + response_length − len(prompt))` (`vllm_async_server.py:564-566`). An over-cap prompt therefore **silently loses response budget**.
- `data.filter_overlong_prompts` defaults to False (`legacy_data.yaml:83`).

**Consequence.**
1. Write a custom reward manager that:
   - sets reward 1 only for a non-truncated, verified answer;
   - logs truncation separately;
   - runs Math-Verify in a subprocess with the 5 s / 15 s timeout and retry rule;
   - raises on an unresolved timeout, so the round is held.
2. Exclude prompts over 1,024 templated tokens before the run.
3. Assert the prompt length per request.
4. Test both cases on fixtures: EOS at exactly the cap, and a cap-hit without EOS.

---

## 13. Companion versions declared by verl 0.9.0

| Package | Declared | Source |
|---|---|---|
| Python | `>=3.10` | `pyproject.toml:24` |
| transformers | `>=5.5.3,!=5.6.0,<5.11` | `setup.py:43`, `requirements.txt` |
| tensordict | `>=0.8.0,<=0.10.0,!=0.9.0` | `setup.py:40` |
| ray[default] | `>=2.41.0` | `setup.py:38` |
| numpy | `>=2.0.0` | `setup.py:32` |
| vllm (extra `vllm`) | `>=0.18.0` | `setup.py:55`; `docs/start/install.rst:25` |
| flash-attn, liger-kernel (extra `gpu`) | unpinned | `setup.py:53` |
| math-verify (extra `math`) | unpinned | `setup.py:54` |
| TransferQueue | `==0.1.8`, in `requirements.txt` only. **Not in `install_requires`**, but V1 imports it (`main_ppo.py:138`, `trainer_base.py:30`) | `requirements.txt` |
| torch | not pinned for vLLM (comes with vLLM); `torch==2.9.1` only in the sglang extra | `setup.py:57-61` |
| Reference image `docker/Dockerfile.stable.vllm` | vllm 0.24.0, torch 2.11.0, CUDA 13.0.2, Python 3.12, flash_attn 2.8.3, TransferQueue 0.1.8, `TRANSFORMERS_VERSION=5.3.0` (conflicts with `setup.py`'s ≥5.5.3; unknown whether the image upgrades it later) | `Dockerfile.stable.vllm:4-22`, `:137-150` |

**Consequence.**
1. Pin vllm, torch, transformers, flash-attn, tensordict, ray and TransferQueue explicitly in our environment.
2. Check the Babel GPU driver against the CUDA version the vLLM wheel needs.
3. Record the resolved versions.

---

## Summary table

| Spec item | verl 0.9.0 default | Setting / override needed | Test needed |
|---|---|---|---|
| A = (r−mean)/(pop_std+1e-6) | `grpo`: Bessel std, eps 1e-6 | Custom `register_adv_est("grpo_popstd")`; `algorithm.adv_estimator=grpo_popstd`; `model.external_lib` | Mixed, all-0 and all-1 groups vs. hand values; check the registered fn is called |
| Zero-variance groups in denominator | Yes (A=0 exactly; fixed `global_batch_size`) | none | Fixture with a zero-variance group |
| Mean over tokens per response, then mean over responses | `token-mean` | `actor.loss_agg_mode=seq-mean-token-mean` | Loss value on fixture |
| Microbatch 1, exact accumulation to 32 | micro-batch unset (required) | `ppo_micro_batch_size_per_gpu=1`, `use_dynamic_bsz=false` | Gradient equality for micro-batch sizes 1/2/4/32 |
| 16 prompts, n=8, 4 steps × 32 responses, 1 epoch | 1024 prompts, n=1, minibatch 256 prompts | `train_batch_size=16`, `rollout.n=8`, `ppo_mini_batch_size=4` (prompts), `ppo_epochs=1` | Count optimizer steps per round = 4; 32 rows each |
| Minibatch = 2 hard + 2 aux complete groups, balanced order | Row order = TQ listing order (unknown); no config | Custom `ReplayBuffer` via `trainer.v1.sampler.custom_sampler`; `actor.shuffle=false` | Log uid/role per minibatch on the real pipeline |
| No silent drop/replace of failed groups | Failed groups kept partial or refilled | Custom sampler asserts 16×8 `finished`, else raises | Inject a failed session; expect a hold |
| Prompt schedule per round | `RandomSampler`, unseeded (`data.seed=null`) | Ordered dataset, `data.shuffle=false` | Round k prompts = schedule rows; also after resume |
| q = π/π_rollout, fixed for 4 steps | π_old = actor recompute once per round | Decide decoupled vs. `bypass_mode=true`; keep `rollout_is`/`rollout_rs` null | Old log prob identical across the 4 minibatches |
| Rollout vs. training log-prob mismatch quantiles | Probability-difference mean/max/std; `rollout_corr/*` KL metrics | Own per-round quantiles of log-prob difference | Tolerance established on the pinned backend |
| min(qA, clip(q,1−εl,1+εh)A), no dual clip | Dual clip on for A<0 (c=3.0, cannot disable); log-ratio clamped ±20 | Custom `register_policy_loss`; `policy_loss.loss_mode=<name>` (fallback `clip_ratio_c=1e10`) | Branch-by-branch fixture |
| Clip fractions by sign | `pg_clipfrac` (mixed), `pg_clipfrac_lower` (= dual-clip fraction) | Emit from the custom loss | Fixture with known fractions |
| No KL, no entropy, no ref model | Already off; ref not built | Set explicitly | Assert no ref worker; no KL/entropy loss terms |
| Dropout off | Train-mode forward; Qwen3 `attention_dropout=0.0` | `model.override_config.attention_dropout=0.0` | No dropout p>0; train-mode = eval-mode logits |
| AdamW (0.9, 0.999), eps 1e-8, wd 0 | wd 0.01; eps implicit 1e-8 | `optim.weight_decay=0`, `override_optimizer_config.eps=1e-8` | Inspect `param_groups` |
| Constant LR, no warmup | constant, warmup 0 | `lr_warmup_steps=0`, `lr_scheduler_type=constant` | LR identical over all steps |
| Grad norm limit 1.0 | `optim.clip_grad=1.0` (`actor.grad_clip` unused) | Set `optim.clip_grad=1.0` | Logged `grad_norm` pre-clip; clipped update norm |
| BF16 compute, FP32 master/states | fp32 params, bf16 FSDP compute + autocast | none | Dtypes of params, grads, Adam states |
| All parameters trainable | Full model, no LoRA by default | `lora_rank=0` | Every param `requires_grad` and changes after a step |
| One GPU, colocated vLLM | `n_gpus_per_node=8`, rollout TP=2, `max_model_len`=40960 | `n_gpus_per_node=1`, TP=1, `max_model_len=5120`, measured `gpu_memory_utilization`, `log_prob_micro_batch_size_per_gpu` | Peak memory at the cap; OOM ladder per spec |
| Sync rollout model after the round | Yes (`on_step_end`) | none | vLLM log probs change after a sync |
| Decoding T=1.0, top_p=1, top_k off, rep. penalty 1.0 | Same (`rollout.yaml:17-23`; `vllm_async_server.py:576`, `:983-993`) | none | Inspect vLLM `SamplingParams` for one request |
| Non-truncated completion required for reward 1 | Not visible to `compute_score` | Custom reward manager (token ids + EOS check) | Cap-with-EOS and cap-without-EOS fixtures |
| Verifier timeout 5 s / retry 15 s, unknown ≠ wrong | No timeout | In the custom reward manager | Hanging-verifier fixture |
| Prompt cap 1,024, never truncate | Not enforced; response budget shrinks silently | Pre-filter; assert per request | Over-cap prompt raises |
| `enable_thinking=False` | Template kwargs empty | `data.apply_chat_template_kwargs.enable_thinking=false` (`legacy_data.yaml:119`) | Rendered-template fixture |
| Resume: optimizer, scheduler, RNG, sampler, counters, logging | Model, optim, sched, actor RNG, dataloader saved; vLLM RNG, uid RNG, W&B id not | `save_freq`; sidecar state; W&B env vars | Forced-interrupt test, no skipped/duplicated round |
| Per-request generation seeds | Only with `full_determinism`, then one shared seed | Leave off; record limitation | If ever enabled: verify the 8 samples differ |
| Version pins | vllm ≥0.18, transformers ≥5.5.3 <5.11, TransferQueue 0.1.8 (not in install_requires) | Pin everything explicitly | Record versions at startup |

Source checkout used for this audit: `/private/tmp/claude-501/-Users-vineethbhat-Desktop-Courses-deep-rl-project-qwen-interference-study-slurm-agentic-engineering/f2158047-8407-4459-95fd-ad10c84c7e2c/scratchpad/verl-src/verl` (tag `v0.9.0`, temporary).
