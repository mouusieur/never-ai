"""Immutable special cognitive, latent-thought, plugin-action, and SEAL self-edit tokens."""

from __future__ import annotations

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"
USER_TOKEN = "<|user|>"
NEVER_TOKEN = "<|never|>"
CONTEXT_TOKEN = "<|context|>"
THINK_START = "<|think|>"
THINK_END = "<|/think|>"
LATENT_BOT = "<|bot|>"
LATENT_EOT = "<|eot|>"
CALL_START = "<|call|>"
CALL_END = "<|/call|>"
RESULT_START = "<|result|>"
RESULT_END = "<|/result|>"
SELF_EDIT_START = "<|self_edit|>"
SELF_EDIT_END = "<|/self_edit|>"

# Strictly ordered IDs 0..15 across all vocabulary morphogenesis stages
SPECIAL_TOKENS: list[str] = [
    PAD_TOKEN,        # 0
    BOS_TOKEN,        # 1
    EOS_TOKEN,        # 2
    USER_TOKEN,       # 3
    NEVER_TOKEN,      # 4
    CONTEXT_TOKEN,    # 5
    THINK_START,      # 6
    THINK_END,        # 7
    LATENT_BOT,       # 8
    LATENT_EOT,       # 9
    CALL_START,       # 10
    CALL_END,         # 11
    RESULT_START,     # 12
    RESULT_END,       # 13
    SELF_EDIT_START,  # 14
    SELF_EDIT_END,    # 15
]

SPECIAL_TOKEN_IDS: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
USER_ID = 3
NEVER_ID = 4
CONTEXT_ID = 5
THINK_START_ID = 6
THINK_END_ID = 7
LATENT_BOT_ID = 8
LATENT_EOT_ID = 9
CALL_START_ID = 10
CALL_END_ID = 11
RESULT_START_ID = 12
RESULT_END_ID = 13
SELF_EDIT_START_ID = 14
SELF_EDIT_END_ID = 15
