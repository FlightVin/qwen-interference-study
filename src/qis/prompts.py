"""Prompt rendering (non-thinking Qwen3 template), decoding constants, vLLM weight sync."""
import torch

USER_PREFIX = ("Solve the following problem. Show your working and put your final answer "
               "in \\boxed{}.\n\n")
PROMPT_CAP = 1024
THINK_STUB = "<think>\n\n</think>\n\n"


def render(tok, problem: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": USER_PREFIX + problem}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def template_fixture(tok) -> dict:
    """Proves the enable_thinking flag is honoured: the non-thinking render ends with the empty
    think block; the thinking render does not."""
    off = render(tok, "What is 1+1?")
    on = tok.apply_chat_template([{"role": "user", "content": USER_PREFIX + "What is 1+1?"}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return {"rendered_nonthinking": off, "rendered_thinking": on,
            "ok": off.endswith("<|im_start|>assistant\n" + THINK_STUB) and not on.endswith(THINK_STUB)}


def vllm_top_k_disabled():
    from vllm import SamplingParams
    return SamplingParams().top_k      # the backend's own "disabled" value, recorded in results


def sync_weights(llm, model) -> str:
    """Copy HF weights (bf16) into the in-process vLLM model. Needs VLLM_ENABLE_V1_MULTIPROCESSING=0."""
    params = [(n, p.detach().to(torch.bfloat16)) for n, p in model.named_parameters()]

    def _load(worker):
        worker.model_runner.model.load_weights(iter(params))
        return True
    llm.collective_rpc(_load)
    return "collective_rpc(load_weights), in-process engine"
