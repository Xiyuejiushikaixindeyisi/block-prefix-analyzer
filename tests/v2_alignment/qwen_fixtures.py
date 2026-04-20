"""V2 alignment fixtures for the Qwen2 target configuration.

Target configuration (config_tag = "qwen2_vllm_aligned")
----------------------------------------------------------
  chat_template : QwenChatTemplate   (Qwen2 <|im_start|> format, Layer 1 VERIFIED)
  tokenizer     : HFTokenizerAdapter("Qwen/Qwen2-7B-Instruct")  (Layer 2 PENDING)
  block_builder : ChainedBlockBuilder with mmh3 prefix-chaining  (Layer 3 PENDING)
  block_size    : 16 (vLLM default for Qwen2 deployments)

Layer-1 rendered_prompt is VERIFIED:
  Derived directly from the official Qwen2 Jinja template published at
  https://huggingface.co/Qwen/Qwen2-7B-Instruct (tokenizer_config.json).
  The QwenChatTemplate class replicates it byte-for-byte.

Layer-2 token_ids is PENDING:
  Requires ``pip install transformers`` and downloading the Qwen2 vocabulary.
  Set expected_token_ids=None until verified with the real tokenizer.

Layer-3 block_ids is PENDING:
  Requires Layer-2 token_ids + ``pip install mmh3`` for the chained MurmurHash3
  used by vLLM's prefix-cache key.  Set expected_block_ids=None until verified.

To fill in pending fields:
  1. pip install transformers mmh3
  2. Run compute_qwen_goldens() (not yet implemented) with the real adapters.
  3. Update expected_token_ids, expected_block_ids, expected_leftover_token_count.
  4. Change alignment_status to "framework_verified".
"""
from __future__ import annotations

from tests.v2_alignment.fixtures import V2AlignmentFixture
from block_prefix_analyzer.v2.schema import Message

# ---------------------------------------------------------------------------
# Qwen2-target fixtures — Layer 1 VERIFIED, Layers 2-3 PENDING
# ---------------------------------------------------------------------------

QWEN_FIXTURE_SINGLE_USER = V2AlignmentFixture(
    name="qwen_single_user",
    description="Single user turn — Qwen2 template format",
    config_tag="qwen2_vllm_aligned",
    alignment_status="pending_framework",
    messages=[Message(role="user", content="Hello")],
    block_size=16,
    # Layer 1: VERIFIED — output matches Qwen2 Jinja template exactly.
    expected_rendered_prompt="<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    # Layers 2-3: PENDING — requires real Qwen2 tokenizer + mmh3.
    expected_token_ids=None,
    expected_block_ids=None,
    expected_leftover_token_count=None,
    framework_notes=(
        "Layer-1 rendered_prompt VERIFIED against Qwen2 Jinja source. "
        "Layers 2-3 pending: install transformers + mmh3 and run with "
        "HFTokenizerAdapter('Qwen/Qwen2-7B-Instruct') + ChainedBlockBuilder."
    ),
)

QWEN_FIXTURE_SYSTEM_USER = V2AlignmentFixture(
    name="qwen_system_user",
    description="System prompt + user turn — Qwen2 <|im_start|> format",
    config_tag="qwen2_vllm_aligned",
    alignment_status="pending_framework",
    messages=[
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ],
    block_size=16,
    expected_rendered_prompt=(
        "<|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    expected_token_ids=None,
    expected_block_ids=None,
    expected_leftover_token_count=None,
    framework_notes=(
        "Layer-1 VERIFIED. Qwen2 system token is <|im_start|>system (not "
        "<|system|>), which changes all block hash values compared to "
        "MinimalChatTemplate. Layers 2-3 pending."
    ),
)

QWEN_FIXTURE_MULTI_TURN = V2AlignmentFixture(
    name="qwen_multi_turn",
    description="4-message dialogue — Qwen2 assistant turn uses <|im_end|>",
    config_tag="qwen2_vllm_aligned",
    alignment_status="pending_framework",
    messages=[
        Message(role="system", content="Be concise."),
        Message(role="user", content="What is 2+2?"),
        Message(role="assistant", content="4"),
        Message(role="user", content="Why?"),
    ],
    block_size=16,
    expected_rendered_prompt=(
        "<|im_start|>system\nBe concise.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n4<|im_end|>\n"
        "<|im_start|>user\nWhy?<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    expected_token_ids=None,
    expected_block_ids=None,
    expected_leftover_token_count=None,
    framework_notes=(
        "Layer-1 VERIFIED. Multi-turn: assistant replies are wrapped in "
        "<|im_start|>assistant\\ncontent<|im_end|>\\n, which is structurally "
        "different from MinimalChatTemplate."
    ),
)

QWEN_FIXTURE_EMPTY_SYSTEM = V2AlignmentFixture(
    name="qwen_empty_system",
    description="System message with empty content — Qwen2 still emits the role header",
    config_tag="qwen2_vllm_aligned",
    alignment_status="pending_framework",
    messages=[
        Message(role="system", content=""),
        Message(role="user", content="Hi"),
    ],
    block_size=16,
    expected_rendered_prompt=(
        "<|im_start|>system\n<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    expected_token_ids=None,
    expected_block_ids=None,
    expected_leftover_token_count=None,
    framework_notes=(
        "Layer-1 VERIFIED. Qwen2 emits empty system block as "
        "<|im_start|>system\\n<|im_end|>\\n — same as MinimalChatTemplate "
        "structurally but different tokens."
    ),
)

# ---------------------------------------------------------------------------
# Exported registry
# ---------------------------------------------------------------------------

#: Qwen2 fixtures whose Layer-1 rendered_prompt is verified.
QWEN_LAYER1_VERIFIED: list[V2AlignmentFixture] = [
    QWEN_FIXTURE_SINGLE_USER,
    QWEN_FIXTURE_SYSTEM_USER,
    QWEN_FIXTURE_MULTI_TURN,
    QWEN_FIXTURE_EMPTY_SYSTEM,
]
