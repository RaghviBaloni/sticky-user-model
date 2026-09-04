"""Stage 2, step 8 -- extract and cache activations (Appendix B).

Pass 2 of the two-pass split (B.1): the frozen transcripts are re-tokenized from scratch and
run teacher-forced through `common.measure_forward`, which re-runs each transcript with every
hook detached and asserts bit-identical logits. No KV cache, no generation-time state.

Positions per dialogue (B.2), at every layer: u_t, eot_u_t for t=1..4 and a_t for t=1..3.
Cache is fp16 .pt shards plus a manifest.parquet sidecar (B.4).
B.3 decode assertion is printed for the first 3 dialogues.
"""

import json
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from common import (  # noqa: E402
    CACHE_ROOT,
    MODEL_ID,
    REPO_ROOT,
    extract_residual_stream,
    find_turn_boundaries,
    load_model_and_tokenizer,
)

TRANSCRIPT_PATH = os.path.join(REPO_ROOT, "data", "transcripts", "stage2_transcripts.json")
CACHE_DIR = os.path.join(CACHE_ROOT, "stage2_transcripts")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.parquet")
ROWS_PER_SHARD = 1024
N_DECODE_CHECK = 3

with open(TRANSCRIPT_PATH) as f:
    payload = json.load(f)
dialogues = payload["dialogues"]
print(f"loaded {len(dialogues)} frozen transcripts from {TRANSCRIPT_PATH}")

model, tokenizer = load_model_and_tokenizer(MODEL_ID)
n_layers = model.config.text_config.num_hidden_layers
hidden_size = model.config.text_config.hidden_size
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"subject model on physical GPU {os.environ['CUDA_VISIBLE_DEVICES']}, "
      f"{n_layers} layers x {hidden_size} hidden")
print(f"cache -> {CACHE_DIR}")

ROLE_SEQUENCE = ["user", "assistant", "user", "assistant", "user", "assistant", "user"]
asserts_fired = []

# ---------------------------------------------------------------------------
# B.3 -- decode every extraction index for the first N dialogues and eyeball it
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"B.3 DECODE ASSERTION -- first {N_DECODE_CHECK} dialogues")
print("=" * 78)
for d in dialogues[:N_DECODE_CHECK]:
    messages = [{"role": r, "content": c} for r, c in zip(
        ROLE_SEQUENCE,
        [d["user_turns"][0], d["assistant_turns"][0], d["user_turns"][1], d["assistant_turns"][1],
         d["user_turns"][2], d["assistant_turns"][2], d["user_turns"][3]])]
    ids = tokenizer(d["transcript_text"], add_special_tokens=False).input_ids
    assert ids == d["transcript_token_ids"], f"{d['dialogue_id']}: tokenization drift"
    boundaries = find_turn_boundaries(tokenizer, ids, messages)
    print(f"\n{d['dialogue_id']}  arm={d['arm']} valences={d['valences']}  ({len(ids)} tokens)")
    for name, idx in boundaries.items():
        print(f"  {name:12s} idx={idx:5d} tok={tokenizer.decode([ids[idx]])!r}")

# ---------------------------------------------------------------------------
# Extraction over all dialogues
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("EXTRACTION")
print("=" * 78)
manifest_rows = []
shard_buffer = []
shard_index = 0
global_row = 0
t0 = time.time()


def flush_shard():
    global shard_buffer, shard_index
    if not shard_buffer:
        return
    tensor = torch.stack(shard_buffer, dim=0)  # (rows, n_layers, hidden)
    path = os.path.join(CACHE_DIR, f"acts_{shard_index:04d}.pt")
    torch.save(tensor, path)
    print(f"  wrote {path} {tuple(tensor.shape)} {tensor.dtype}")
    shard_buffer = []
    shard_index += 1


for i, d in enumerate(dialogues):
    messages = [{"role": r, "content": c} for r, c in zip(
        ROLE_SEQUENCE,
        [d["user_turns"][0], d["assistant_turns"][0], d["user_turns"][1], d["assistant_turns"][1],
         d["user_turns"][2], d["assistant_turns"][2], d["user_turns"][3]])]
    ids = tokenizer(d["transcript_text"], add_special_tokens=False).input_ids
    if ids != d["transcript_token_ids"]:
        asserts_fired.append(f"{d['dialogue_id']}: tokenization drift between passes")
        raise AssertionError(f"{d['dialogue_id']}: tokenization drift between passes")
    boundaries = find_turn_boundaries(tokenizer, ids, messages)

    ids_t = torch.tensor([ids], device="cuda")
    resid = extract_residual_stream(model, ids_t)  # (n_layers, seq, hidden) fp16
    assert resid.shape == (n_layers, len(ids), hidden_size), resid.shape

    for name, idx in boundaries.items():
        role_kind, turn_str = name.rsplit("_", 1)
        turn_index = int(turn_str)
        if name.startswith("eot_u"):
            token_role, label_turn = "eot_u", turn_index
        elif name.startswith("u"):
            token_role, label_turn = "u", turn_index
        else:
            token_role, label_turn = "a", turn_index
        # user positions carry that turn's valence; assistant positions carry the valence
        # of the user turn they are responding to.
        label = d["valences"][label_turn - 1]
        shard_buffer.append(resid[:, idx, :].cpu())
        manifest_rows.append(dict(
            dialogue_id=d["dialogue_id"], condition=d["arm"], topic=d["topic"],
            phrasing=d["phrasing"], replicate=d["replicate"], turn_index=turn_index,
            token_role=token_role, position_name=name, label=label,
            token_index=idx, n_tokens=len(ids), row_offset=global_row,
            shard=shard_index, row_in_shard=len(shard_buffer) - 1))
        global_row += 1
        if len(shard_buffer) >= ROWS_PER_SHARD:
            flush_shard()

    if (i + 1) % 80 == 0:
        print(f"  {i + 1}/{len(dialogues)} dialogues, {time.time() - t0:.0f}s")

flush_shard()
manifest = pd.DataFrame(manifest_rows)
manifest.to_parquet(MANIFEST_PATH)
print(f"\nextracted {len(manifest)} rows from {len(dialogues)} dialogues "
      f"in {time.time() - t0:.0f}s")
print(f"manifest -> {MANIFEST_PATH}")
print(f"cache size: "
      f"{sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR)) / 1e9:.2f} GB")
print(f"\nrows per position:\n{manifest.token_role.value_counts().to_string()}")
print(f"\nrows per arm:\n{manifest.condition.value_counts().to_string()}")
print(f"\nasserts fired: {len(asserts_fired)}")
print("\nSTEP 8 COMPLETE")
