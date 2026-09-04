"""Shared utilities. Import from here; do not re-implement these in stage scripts.

Two of these functions exist specifically to make a silent, result-corrupting failure
impossible rather than merely unlikely:

  * `measure_forward()`   -- every measurement pass re-runs itself with all hooks detached
                             and asserts bit-identical logits (ROADMAP Appendix C.4).
  * `find_turn_boundaries()` -- asserts the role token after each <|im_start|> against the
                             role declared in `messages`. Roles are never inferred from
                             odd/even position (Appendix B.2/B.3).

Model facts verified on Qwen/Qwen3.5-4B (Stage 0, see notes in highlights.md):
  * decoder layers live at `model.model.language_model.layers` and return a bare Tensor
    from forward(), not a `(hidden_states, ...)` tuple;
  * `output_hidden_states=True` gives n_layers+1 entries, `hidden_states[l+1]` is the
    output of layer `l` (verified equal to a raw forward hook's capture);
  * two identical forward passes are bit-identical, so `torch.equal` needs no tolerance.
"""

import contextlib
import dataclasses
import json
import os

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"          # subject model: probed and steered
GENERATOR_MODEL_ID = "Qwen/Qwen3.5-9B"  # turn-pool generator only; never probed

# QC-gate sentiment classifier (Appendix A.5). NOTE: metrics.md Sec 3's "off-the-shelf
# sentiment classifier" slot is still blank -- locking that choice is the human's call.
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_PATH = os.path.join(REPO_ROOT, "data", "pool", "turn_pool.json")

# Activation caches do not fit in the home quota (110 GB hard, ~5 GB free), so the cache
# lives on the big data volume and repo `data/cache` is a symlink to it.
CACHE_ROOT = os.environ.get(
    "STICKY_CACHE_ROOT", "/media/data/baloni/sticky-user-model/cache"
)

# Human-supplied constants, cross-checked against the tokenizer at runtime in
# resolve_special_ids(). They are asserted, not trusted: if the tokenizer ever disagrees
# the run dies loudly instead of mislabelling turns.
EXPECTED_ROLE_TOKEN_IDS = {"user": 846, "assistant": 74455}


# The home quota (110 GB hard) is effectively full, so not every model fits in ~/.cache.
# Complete snapshots on the big /media/data volume are used read-only when present.
EXTRA_CACHE_ROOTS = ("/media/data/.cache/huggingface/hub",)


def _snapshot_is_complete(snapshot_dir):
    """True only if every shard named by the weight index is present and non-empty.

    Guards against resolving a partially-downloaded snapshot: a truncated shard set would
    otherwise load as a plausible-looking model or die deep inside from_pretrained.
    """
    shards = None
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = os.path.join(snapshot_dir, index_name)
        if os.path.exists(index_path):
            with open(index_path) as f:
                shards = set(json.load(f)["weight_map"].values())
            break
    if shards is None:
        present = [
            name for name in ("model.safetensors", "pytorch_model.bin")
            if os.path.exists(os.path.join(snapshot_dir, name))
        ]
        if not present:
            return False
        shards = {present[0]}
    for shard in shards:
        path = os.path.join(snapshot_dir, shard)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
    return True


def resolve_model_path(model_id):
    """Local snapshot dir for `model_id` if a complete one exists, else `model_id` itself."""
    if os.path.isdir(model_id):
        return model_id
    home_hub = os.path.join(
        os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")), "hub"
    )
    for root in [home_hub, *EXTRA_CACHE_ROOTS]:
        snapshots = os.path.join(root, "models--" + model_id.replace("/", "--"), "snapshots")
        if not os.path.isdir(snapshots):
            continue
        for snap in sorted(os.listdir(snapshots)):
            candidate = os.path.join(snapshots, snap)
            if os.path.exists(os.path.join(candidate, "config.json")) and _snapshot_is_complete(candidate):
                return candidate
    return model_id


def load_model_and_tokenizer(model_id=MODEL_ID, device="cuda", dtype=torch.bfloat16):
    """Load a model on one GPU. HF only -- no serving layer (CLAUDE.md rule 8)."""
    path = resolve_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def resolve_special_ids(tokenizer):
    """Resolve chat-markup and role token ids from the tokenizer, asserting by decoding."""
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert im_start is not None and im_end is not None, "could not resolve <|im_start|>/<|im_end|>"

    roles = {}
    for role in ("system", "user", "assistant"):
        enc = tokenizer(role, add_special_tokens=False).input_ids
        assert len(enc) == 1, f"role {role!r} is not a single token: {enc}"
        assert tokenizer.decode(enc) == role, f"role token {enc[0]} does not decode back to {role!r}"
        roles[role] = enc[0]

    for role, expected in EXPECTED_ROLE_TOKEN_IDS.items():
        assert roles[role] == expected, (
            f"role token id for {role!r} is {roles[role]}, expected {expected} -- "
            "tokenizer changed under us; re-verify EXPECTED_ROLE_TOKEN_IDS before running anything"
        )

    return {"im_start": im_start, "im_end": im_end, "roles": roles}


def generation_kwargs(tokenizer, max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True):
    """Generation settings shared by every generation pass.

    NOTE: Qwen/Qwen3.5-4B ships no generation_config.json, so the model's default
    eos_token_id is <|endoftext|> (248044) and NOT the chat turn terminator <|im_end|>
    (248046). Without setting this explicitly every generated turn runs to the token cap
    instead of stopping at the end of the turn. Set here so no caller can forget.
    """
    special = resolve_special_ids(tokenizer)
    return dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=[special["im_end"], tokenizer.convert_tokens_to_ids("<|endoftext|>")],
        pad_token_id=tokenizer.pad_token_id,
    )


# ---------------------------------------------------------------------------
# Turn pool: the only legal source of dialogue content (ROADMAP Sec 5a, Appendix A.2)
# ---------------------------------------------------------------------------
TOPICS = ("travel", "health", "work", "tech_support")
PHRASINGS = ("a", "b")
VALENCES = ("S", "N", "H")  # distressed / neutral / positive
SLOTS = (1, 2, 3)
CHAT_MARKUP = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>")


@dataclasses.dataclass(frozen=True)
class PoolTurn:
    """One generated user turn, carrying its full provenance.

    Every piece of dialogue content that enters the pipeline is one of these. Raw dicts
    are rejected at the transcript-builder boundary by require_pool_turns() -- the
    hand-written DIALOGUE / DIALOGUE_NEUTRAL fixtures in 00_infra_gate.py are smoke-test
    scaffolding, not data, and this is what keeps them out of a result.

    kind="user"  -> valence in {S,N,H}, slot in {1,2,3}: U[topic, phrasing, valence, slot]
    kind="final" -> valence "N", slot 4: F[topic, phrasing], the shared final neutral turn
                    that must be byte-identical across conditions within a family (A.3).
    """

    text: str
    kind: str
    topic: str
    phrasing: str
    valence: str
    slot: int
    generator_model: str = ""
    seed: int = -1
    temperature: float = 0.0
    top_p: float = 0.0
    attempt: int = 1
    source: str = "generated"
    note: str = ""  # provenance for any human-directed edit or replacement

    def __post_init__(self):
        assert self.kind in ("user", "final"), f"bad kind {self.kind!r}"
        assert self.source in ("generated", "hand_authored"), f"bad source {self.source!r}"
        if self.source == "generated":
            assert self.generator_model, "generated turns must name their generator model"
        assert self.topic in TOPICS, f"bad topic {self.topic!r}"
        assert self.phrasing in PHRASINGS, f"bad phrasing {self.phrasing!r}"
        assert self.valence in VALENCES, f"bad valence {self.valence!r}"
        if self.kind == "user":
            assert self.slot in SLOTS, f"user turn slot must be in {SLOTS}, got {self.slot}"
        else:
            assert self.valence == "N", "final turns are neutral by construction"
            assert self.slot == 4, f"final turn slot must be 4, got {self.slot}"
        assert self.text and self.text == self.text.strip(), "text must be non-empty and stripped"
        assert "\n" not in self.text, "pool turns are single-paragraph"
        for marker in CHAT_MARKUP:
            assert marker not in self.text, f"pool turn contains chat markup {marker!r}"

    @property
    def turn_id(self):
        if self.kind == "final":
            return f"F[{self.topic},{self.phrasing}]"
        return f"U[{self.topic},{self.phrasing},{self.valence},{self.slot}]"

    def as_message(self):
        """Render as a chat message. The only sanctioned turn -> message conversion."""
        return {"role": "user", "content": self.text}


def require_pool_turns(turns, where="transcript builder"):
    """Boundary guard: only PoolTurn objects may cross into dialogue assembly.

    Call this as the first statement of anything that builds a transcript. A hand-written
    dict looks exactly like real data once it is inside a message list, and would be
    indistinguishable in the cache and in the manifest -- so it is refused here, by type,
    rather than caught later by eye.
    """
    turns = list(turns)
    for i, turn in enumerate(turns):
        if not isinstance(turn, PoolTurn):
            raise TypeError(
                f"{where}: item {i} is {type(turn).__name__}, not PoolTurn. Dialogue content "
                "must come from the generated turn pool (data/pool/turn_pool.json). "
                "Hand-written fixtures are smoke-test scaffolding and must not enter the pipeline."
            )
    return turns


def save_turn_pool(turns, path=POOL_PATH, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "meta": meta or {},
        "turns": [dataclasses.asdict(t) for t in require_pool_turns(turns, where="save_turn_pool")],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def load_turn_pool(path=POOL_PATH):
    """Load the pool back as typed objects, re-running every PoolTurn validation."""
    with open(path) as f:
        payload = json.load(f)
    turns = [PoolTurn(**row) for row in payload["turns"]]
    return turns, payload.get("meta", {})


def load_sentiment_scorer(device="cpu"):
    """Off-the-shelf 3-class sentiment classifier, for the Appendix A.5 QC gate.

    Returns score(texts) -> list of {negative, neutral, positive, compound}, where
    compound = P(positive) - P(negative) in [-1, 1]. A QC-gate measure; metrics.md Sec 3
    is not locked to this model.
    """
    from transformers import AutoModelForSequenceClassification

    path = resolve_model_path(SENTIMENT_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    model.eval()
    label_order = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    assert set(label_order) == {"negative", "neutral", "positive"}, label_order

    def score(texts):
        out = []
        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=512).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)
            for row in probs:
                d = {label: row[i].item() for i, label in enumerate(label_order)}
                d["compound"] = d["positive"] - d["negative"]
                out.append(d)
        return out

    return score


# ---------------------------------------------------------------------------
# Patch 2 -- boundary scanner with role assertion
# ---------------------------------------------------------------------------
def find_turn_boundaries(tokenizer, ids, messages):
    """Locate u_t / eot_u_t / a_t for every turn (Appendix B.2).

    `messages` is the message list the token sequence was rendered from; the expected role
    of each turn comes from it. Roles are NEVER inferred from odd/even position -- a
    dropped or duplicated turn (e.g. an assistant generation that came back empty and got
    discarded) would silently shift every subsequent label under a parity assumption, and
    the resulting activations would be mislabelled but perfectly plausible-looking.

    Asserts, in order:
      * one <|im_start|> and one <|im_end|> per message (content containing chat markup
        breaks this loudly instead of corrupting the scan);
      * markers do not interleave across messages;
      * the token immediately after each <|im_start|> is the role token declared in
        `messages` for that message;
      * every message has a non-empty content span.

    Returns {"u_1": idx, "eot_u_1": idx, "a_1": idx, ...}: u_t/a_t are the last *content*
    token of that turn, eot_u_t is the <|im_end|> that closes user turn t. System messages
    are role-asserted but contribute no readout position.
    """
    ids = list(ids)
    special = resolve_special_ids(tokenizer)
    im_start, im_end, role_ids = special["im_start"], special["im_end"], special["roles"]

    starts = [i for i, t in enumerate(ids) if t == im_start]
    ends = [i for i, t in enumerate(ids) if t == im_end]
    assert len(starts) == len(messages), (
        f"found {len(starts)} <|im_start|> markers for {len(messages)} messages -- "
        "message content probably contains chat markup, or the chat template changed"
    )
    assert len(ends) == len(messages), (
        f"found {len(ends)} <|im_end|> markers for {len(messages)} messages -- "
        "message content probably contains chat markup, or the chat template changed"
    )

    boundaries = {}
    counts = {"system": 0, "user": 0, "assistant": 0}
    for i, (start, end, msg) in enumerate(zip(starts, ends, messages)):
        role = msg["role"]
        assert role in role_ids, f"message {i}: unknown role {role!r}"
        assert start < end, (
            f"message {i}: <|im_start|> at {start} does not precede its <|im_end|> at {end}"
        )
        if i + 1 < len(starts):
            assert end < starts[i + 1], (
                f"message {i}: <|im_end|> at {end} falls after the next <|im_start|> at "
                f"{starts[i + 1]} -- turn markers interleave"
            )

        role_tok = ids[start + 1]
        assert role_tok == role_ids[role], (
            f"message {i}: expected role {role!r} (token id {role_ids[role]}) after "
            f"<|im_start|> at index {start}, found token id {role_tok} = "
            f"{tokenizer.decode([role_tok])!r} -- declared roles do not match the rendered "
            "transcript; turn labels would be wrong"
        )
        assert end - 1 > start + 1, f"message {i}: empty content span between {start} and {end}"

        counts[role] += 1
        t = counts[role]
        if role == "user":
            boundaries[f"u_{t}"] = end - 1
            boundaries[f"eot_u_{t}"] = end
        elif role == "assistant":
            boundaries[f"a_{t}"] = end - 1

    return boundaries


# ---------------------------------------------------------------------------
# Patch 1 -- empirical intervention-leak guard on every measurement pass
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def all_hooks_detached(model):
    """Temporarily strip every forward / forward-pre hook from every submodule.

    Used to build a reference path that provably cannot be touched by a steering or
    ablation hook. This detaches nnsight's own bookkeeping hooks too, which is fine --
    outside a trace context they only record, they do not modify activations.
    """
    saved = []
    for module in model.modules():
        saved.append((module, module._forward_hooks.copy(), module._forward_pre_hooks.copy()))
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
    try:
        yield
    finally:
        for module, fwd, pre in saved:
            module._forward_hooks.clear()
            module._forward_hooks.update(fwd)
            module._forward_pre_hooks.clear()
            module._forward_pre_hooks.update(pre)


def measure_forward(model, input_ids, attention_mask=None, output_hidden_states=False):
    """The only sanctioned measurement path. Every stage's measurement goes through here.

    Runs the measurement forward pass, then re-runs the identical input through a path
    with every hook detached, and asserts the logits are bit-identical. A steering or
    ablation hook still attached at measurement time changes the first pass and not the
    second, so the assert fires -- which is the difference between measuring a loop and
    measuring the intervention itself (Appendix C.4/C.6, the fatal failure mode).

    `len(model._active_hooks) == 0` cannot be used for this: NNsight registers its own
    bookkeeping forward hooks on every module, so a zero-hook check never holds and would
    have to be disabled -- which is how the guard gets quietly dropped. This check is
    empirical instead: it tests the thing that actually matters, the numbers.

    KV cache is disabled on both passes: measurement is teacher-forced over the full
    transcript and must never reuse generation-time state (Appendix C.6).

    Costs exactly one extra forward pass: measured 2.00x on this model (290-token
    transcript: 191 ms -> 384 ms; 600-token: 222 ms -> 446 ms), i.e. ~62 s rather than
    ~31 s to measure 160 transcripts one at a time. Batch the measurement pass if that
    ever matters.
    """
    kwargs = dict(input_ids=input_ids, use_cache=False, output_hidden_states=output_hidden_states)
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        out_measure = model(**kwargs)
    with torch.no_grad(), all_hooks_detached(model):
        out_clean = model(**kwargs)

    assert torch.equal(out_measure.logits, out_clean.logits), \
        "intervention leaked into measurement pass"

    return out_measure


def extract_residual_stream(model, input_ids, attention_mask=None, dtype=torch.float16):
    """Residual stream at every layer for one transcript, via the leak-guarded path.

    Returns (n_layers, seq_len, hidden) -- `hidden_states[l + 1]` is the output of layer
    `l`; `hidden_states[0]` (the embedding output) is dropped.
    """
    out = measure_forward(model, input_ids, attention_mask=attention_mask, output_hidden_states=True)
    n_layers = model.config.text_config.num_hidden_layers
    assert len(out.hidden_states) == n_layers + 1, (
        f"expected {n_layers + 1} hidden_states, got {len(out.hidden_states)}"
    )
    resid = torch.stack(out.hidden_states[1:], dim=0)  # (n_layers, batch, seq, hidden)
    assert resid.shape[1] == 1, "extract_residual_stream expects a single transcript"
    return resid.squeeze(1).to(dtype)
