"""Rollout engine: in-process vLLM (BF16), sleep mode between generation and training.

Decoding (project.md Section 6): temperature 1.0, top_p 1.0, top_k disabled (backend default),
repetition penalty 1.0, max_tokens = response cap. One request per prompt with n samples and a
per-request seed from a named stream; the seed mapping is returned for the attempt records.
Requires VLLM_ENABLE_V1_MULTIPROCESSING=0 so weights can be loaded in place.
"""
import os

import torch

from qis import prompts


class Rollout:
    def __init__(self, model_path: str, max_tokens: int, gpu_mem_util: float, max_num_seqs: int = 256,
                 max_model_tokens: int = None):
        """max_tokens: default response cap; max_model_tokens: largest cap this engine must serve (A10)."""
        assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "need in-process vLLM"
        from vllm import LLM
        self.max_tokens = max_tokens
        self.max_model_len = prompts.PROMPT_CAP + max(max_tokens, max_model_tokens or 0)
        self.llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=gpu_mem_util,
                       enable_sleep_mode=True, max_model_len=self.max_model_len,
                       seed=0, enable_prefix_caching=False, max_num_seqs=max_num_seqs)
        self.awake = True

    def sampling(self, n: int, seed: int, logprobs: bool, max_tokens: int = None, min_tokens: int = 0):
        from vllm import SamplingParams
        cap = max_tokens or self.max_tokens
        assert prompts.PROMPT_CAP + cap <= self.max_model_len, "cap exceeds the engine context"
        return SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=prompts.vllm_top_k_disabled(),
                              repetition_penalty=1.0, max_tokens=cap, min_tokens=min_tokens,
                              logprobs=0 if logprobs else None, seed=seed)

    def generate(self, prompt_ids: list, n: int, req_seeds: list, logprobs: bool = True,
                 max_tokens: int = None, min_tokens=0):
        """prompt_ids: list of token-id lists. Returns per prompt a list of n dicts:
        ids, finish ('stop'|'length'), lp (sampling log probs or None)."""
        self.wake()
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompt_ids],
                                 [self.sampling(n, s, logprobs, max_tokens,
                                                min_tokens[i] if isinstance(min_tokens, list) else min_tokens)
                                  for i, s in enumerate(req_seeds)], use_tqdm=False)
        res = []
        for o in outs:
            group = []
            for c in o.outputs:
                lp = [d[t].logprob for d, t in zip(c.logprobs, c.token_ids)] if logprobs else None
                group.append({"ids": list(c.token_ids), "finish": c.finish_reason, "lp": lp,
                              "stop_reason": c.stop_reason})
            res.append(group)
        return res

    def sleep(self):
        if self.awake:
            self.llm.sleep(level=1)
            torch.cuda.empty_cache()
            self.awake = False

    def wake(self):
        if not self.awake:
            self.llm.wake_up()
            self.awake = True

    def load_weights(self, model):
        self.wake()
        return prompts.sync_weights(self.llm, model)
