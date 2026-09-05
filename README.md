# Sticky User Models

Does an LLM's own generated response feed back into its internal model of the user?

MATS 10.0 application task — Neel Nanda stream.

## Start here

| File | Purpose |
|---|---|
| `ROADMAP.md` | Full research spec: hypotheses, execution plan, appendices A–D |
| `CLAUDE.md` | Rules for the coding agent (read automatically by Claude Code) |
| `notes/positioning.md` | Literature positioning (Stage 1 output) |
| `notes/highlights.md` | Running results log |
| `notes/red_team.md` | Self-criticism (Stage 5) |
| `notes/timelog.md` | Hour tracking against the 16h/20h budget |

## Layout

```
src/                  numbered scripts, one per experiment
data/pool/            generated turn pool (~240 turns)
data/transcripts/     frozen conversation transcripts (generation pass output)
data/cache/           fp16 activations + manifest.parquet
results/figures/      figures for the writeup
results/tables/       numeric results
notes/                logs, positioning, red-team
```

## Hardware

4× NVIDIA L40S (46 GB, PCIe, no NVLink). Data-parallel — one model per GPU via
`CUDA_VISIBLE_DEVICES`. Do not shard small models.

## The one thing to remember

Generation and measurement are separate passes. Generation caches nothing; a second teacher-forced
forward pass over the frozen transcript produces every measured activation.
