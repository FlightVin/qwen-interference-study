"""Token limits shared by training and evaluation (no heavy imports, so experiment generators can use them)."""

MAX_TOKENS = 12288  # response cap in tokens; an answer that reaches it is truncated and earns reward 0
