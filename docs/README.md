# ScrapMem

## Paper

**ScrapMem: A Bio-Inspired Framework for On-Device Personalized Agent Memory via Optical Forgetting**

- [arXiv:2605.03804](https://arxiv.org/abs/2605.03804)

## Environment

Python 3.10+ with CUDA for the 4-bit Qwen models, and `ffmpeg`/`ffprobe` for video ingestion.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Models used by the paper configuration:

| Role       | Model                  |
| ---------- | ---------------------- |
| Perception | Qwen3-VL-2B-Instruct   |
| Answering  | Qwen3-VL-8B-Instruct   |
| Embedding  | all-MiniLM-L6-v2       |

Both Qwen models use 4-bit quantization on CUDA.
