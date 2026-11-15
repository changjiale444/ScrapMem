# ScrapMem

**ScrapMem: A Bio-Inspired Framework for On-Device Personalized Agent Memory via Optical Forgetting**  
Jiale Chang · Yuxiang Ren  
KDD 2027 Research Track · Volume 1 · [Paper](https://arxiv.org/abs/2605.03804)

ScrapMem consolidates daily multimodal records into Scrapbook Pages. Optical
perception produces text describing visual evidence; nodes extracted from that text
are linked with original-text nodes into daily EM-Paths. Retrieval first matches
query nodes against the daily EM-Graph through a sparse Q-matrix, then reranks
fine-grained evidence records inside the retrieved days for model input.

## Installation

Python 3.10+, CUDA for the 4-bit Qwen models, and `ffmpeg`/`ffprobe` for video.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The paper configuration uses Qwen3-VL-2B-Instruct for perception, Qwen3-VL-8B-Instruct
for answering, and all-MiniLM-L6-v2 for embeddings.

## Experiments

### ATM-Bench (Tables 1 and 3)

```bash
python scripts/run_atm.py --data-root "$DATA_ROOT" \
  --store stores/atm-no-forget --output outputs/atm-no-forget \
  --config configs/paper.json --k 10 --judge-config configs/judge.example.json
```

For a forgetting condition, use a fresh store/output and add
`--policy timed_gentle --as-of YYYY-MM-DD`.

`$DATA_ROOT` points to a directory containing `raw_memory/`, `processed_memory/`,
and `atm-bench/`, obtained from the [ATM-Bench project](https://atmbench.github.io/).

Number and List Recall answers are scored with the ATM-Bench deterministic normalizer
and Jaccard scorer; Open-ended answers use the GPT-5-mini binary judge
(`OPENAI_API_KEY` required).

### Baselines (Table 1)

```bash
python scripts/run_baseline.py --name Mem0 --upstream /path/to/ATM-Bench \
  --data-root "$DATA_ROOT" --k 10
```

For Self-RAG or other official entry points, pass `--command-json` with the exact
argument list required by that implementation.

### Controlled experiments (Appendix B)

```bash
python scripts/run_controlled.py --records data/controlled/records.json \
  --questions data/controlled/questions.json --output outputs/controlled-full \
  --config configs/controlled.example.json --variant full --as-of YYYY-MM-DD
```

`configs/controlled-generation.example.json` generates the 10-user / 1,100-page
synthetic corpus used in Appendix B.

## Library

`src/scrapmem/` is importable as a package, and `python -m scrapmem` exposes the
same pipeline (ingest, retrieve, answer, forget, run-atm, run-baseline,
controlled-*).

## Structure

- `src/scrapmem/` — the ScrapMem library
- `scripts/` — experiment entry points
- `configs/` — model and memory configuration
- `docs/` — paper link and environment setup

## Citation

See `CITATION.cff`. The ACM DOI will be added after publication.

## License

MIT. Adapted ATM-Bench evaluation utilities retain their upstream MIT notice under
`src/scrapmem/_vendor/atmbench/`.
