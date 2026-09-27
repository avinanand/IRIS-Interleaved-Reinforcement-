# IRIS

### Interleaved Reinforcement with Incremental Staged Curriculum for Cross-Lingual Mathematical Reasoning

**Published at ACL 2026 · Main Conference · Long Paper**  
[Paper](https://aclanthology.org/2026.acl-long.1017/) · [PDF](https://aclanthology.org/2026.acl-long.1017.pdf) · [Training guide](docs/training.md)

Official implementation of **IRIS**.

> **TL;DR:** IRIS teaches mathematical reasoning along two axes: solving progressively harder problems and completing solutions with progressively less guidance. Interleaving supervised fine-tuning with reinforcement learning strengthens cross-lingual reasoning, especially in Hindi and Marathi.


IRIS combines two complementary curricula:

- **Vertical axis — problem difficulty:** supervised fine-tuning builds reasoning skills on increasingly challenging problems.
- **Horizontal axis — reasoning independence:** reverse-curriculum reinforcement learning progressively removes supplied solution steps, teaching the model to complete more of the reasoning itself.

The framework interleaves these axes and uses **GRPO** with rewards for correctness, step-wise alignment, continuity, and answer formatting. The paper evaluates IRIS across mathematical reasoning benchmarks and multilingual settings; see the [paper](https://aclanthology.org/2026.acl-long.1017/) for the full method and results.

## Code

| Script | Purpose |
| --- | --- |
| [`VerticalAxis_SFT.py`](VerticalAxis_SFT.py) | Supervised LoRA fine-tuning; default backbone: Qwen2.5-Math-7B |
| [`HorizontalAxis_GRPO.py`](HorizontalAxis_GRPO.py) | Staged reasoning continuations, curriculum sampling, and GRPO training |
| [`Data_Augment.py`](Data_Augment.py) | Generate step-wise solutions using Together AI |
| [`Translation.py`](Translation.py) | Translate training examples into Hindi or Marathi using IndicTrans2 |
| [`Inference.py`](Inference.py) | Generate responses from a trained adapter |

The scripts provide the individual training stages. Configure checkpoint handoffs and the interleaving schedule explicitly for your experiment.

## Getting started

Use Python 3.10/3.11 and a CUDA-enabled Linux environment for training. Create **separate environments for SFT and GRPO**, since the scripts use different TRL interfaces.

```bash
# SFT environment
python3.11 -m venv .venv-sft
source .venv-sft/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/sft.txt
```

Set the paths in `VerticalAxis_SFT.py`, then run:

```bash
python VerticalAxis_SFT.py
```

For GRPO, install `requirements/grpo.txt` in a separate environment, set `DATASET_DIR`, `SFT_CHECKPOINT_PATH`, and `OUTPUT_DIR` in `HorizontalAxis_GRPO.py`, and also update `GRPOConfig.output_dir` to match:

```bash
python HorizontalAxis_GRPO.py
```

SFT expects `DATASET_PATH` to be a JSON **file**; GRPO expects a directory of JSON files. The dependency profiles are starting points, not a validated reproduction environment. See the [training guide](docs/training.md) for complete setup, data preparation, inference, and compatibility notes.

## Data

IRIS introduces **CL-Math**, with step-level mathematical reasoning annotations in English, Hindi, and Marathi. A small repository release of **500 selected examples per language** will be released.

You can also use your own data: provide JSON records with `question`, `answer`, and `step_wise_answer`, keeping numbered solution steps on separate lines. See the [data formats](docs/training.md#data-formats) for a minimal example.

## Citation

If you use IRIS in your research, please cite:

```bibtex
@inproceedings{gupta2026iris,
  title={IRIS: Interleaved Reinforcement with Incremental Staged Curriculum for Cross-Lingual Mathematical Reasoning},
  author={Gupta, Navya and Vyalla, Rishitej Reddy and Anand, Avinash and Kirtani, Chhavi and Cambria, Erik and Zhang, Zhengchen and Wang, Zhengkui and Liu, Timothy and Ng, Aik Beng and See, Simon and Shah, Rajiv Ratn},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={22216--22248},
  year={2026},
  url={https://aclanthology.org/2026.acl-long.1017/},
  doi={10.18653/v1/2026.acl-long.1017}
}
```

## License

The code is released under the [MIT License](LICENSE).
