# Prompt Depth Anything with Rerun
An unofficial implementation of Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation. Using the high resolution depth maps for 3D reconstruction and traversability cost map generation.

Uses [Rerun](https://rerun.io/) to visualize and [uv](https://docs.astral.sh/uv/) for dependency management.

<p align="center">
  <a title="Website" href="https://rerun.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
          <img src="https://img.shields.io/badge/Rerun-0.21.0-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzQ0MV8xMTAzOCkiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSI4IiBmaWxsPSJibGFjayIvPgo8cGF0aCBkPSJNMy41OTcwMSA1Ljg5NTM0TDkuNTQyOTEgMi41MjM1OUw4Ljg3ODg2IDIuMTQ3MDVMMi45MzMgNS41MTg3NUwyLjkzMjk1IDExLjI5TDMuNTk2NDIgMTEuNjY2MkwzLjU5NzAxIDUuODk1MzRaTTUuMDExMjkgNi42OTc1NEw5LjU0NTc1IDQuMTI2MDlMOS41NDU4NCA0Ljk3NzA3TDUuNzYxNDMgNy4xMjI5OVYxMi44OTM4SDcuMDg5MzZMNi40MjU1MSAxMi41MTczVjExLjY2Nkw4LjU5MDY4IDEyLjg5MzhIOS45MTc5NUw2LjQyNTQxIDEwLjkxMzNWMTAuMDYyMUwxMS40MTkyIDEyLjg5MzhIMTIuNzQ2M0wxMC41ODQ5IDExLjY2ODJMMTMuMDM4MyAxMC4yNzY3VjQuNTA1NTlMMTIuMzc0OCA0LjEyOTQ0TDEyLjM3NDMgOS45MDAyOEw5LjkyMDkyIDExLjI5MTVMOS4xNzA0IDEwLjg2NTlMMTEuNjI0IDkuNDc0NTRWMy43MDM2OUwxMC45NjAyIDMuMzI3MjRMMTAuOTYwMSA5LjA5ODA2TDguNTA2MyAxMC40ODk0TDcuNzU2MDEgMTAuMDY0TDEwLjIwOTggOC42NzI1MlYyLjk5NjU2TDQuMzQ3MjMgNi4zMjEwOUw0LjM0NzE3IDEyLjA5Mkw1LjAxMDk0IDEyLjQ2ODNMNS4wMTEyOSA2LjY5NzU0Wk05LjU0NTc5IDUuNzMzNDFMOS41NDU4NCA4LjI5MjA2TDcuMDg4ODYgOS42ODU2NEw2LjQyNTQxIDkuMzA5NDJWNy41MDM0QzYuNzkwMzIgNy4yOTY0OSA5LjU0NTg4IDUuNzI3MTQgOS41NDU3OSA1LjczMzQxWiIgZmlsbD0id2hpdGUiLz4KPC9nPgo8ZGVmcz4KPGNsaXBQYXRoIGlkPSJjbGlwMF80NDFfMTEwMzgiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==">
      </a>
    <a title="Website" href="https://promptda.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
    <a title="arXiv" href="https://arxiv.org/abs/2412.14015" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
    </a>
    <a title="Github" href="https://github.com/rerun-io/prompt-da" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/rerun-io/prompt-da?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
  </p>

<p align="center">
  <img src="media/promptda-github-demo.gif" alt="example output" width="720" />
</p>

## Installation

Make sure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
git clone https://github.com/rerun-io/prompt-da.git
cd prompt-da
uv sync
```

## Usage

### CLI

Run inference on a Polycam zip file:

```bash
uv run polycam-prompt-da --polycam-zip-path data/6G-room-example.zip
```

With the Rerun web viewer:

```bash
uv run polycam-prompt-da --polycam-zip-path data/6G-room-example.zip --rr-config.serve
```

Save to an `.rrd` file for later viewing:

```bash
uv run polycam-prompt-da --polycam-zip-path data/6G-room-example.zip --rr-config.save output.rrd
uv run rerun output.rrd --web-viewer
```

### Cost Map

A 2D traversability cost map is generated from the reconstructed mesh after all frames are processed. The cost map is a bird's-eye-view grid projected onto the XZ ground plane, where each cell encodes traversability:

- **Green** (cost 0.0) = flat, traversable ground
- **Red** (cost 1.0) = obstacle, steep surface, or unobserved

Cost is computed from three components:
- **Step height** (40%) — height range within a cell vs `--max-step-height`
- **Roughness** (30%) — height standard deviation (uneven terrain)
- **Slope** (30%) — surface normal deviation from vertical

Tunable parameters:

| Flag | Default | Description |
|------|---------|-------------|
| `--cost-map-resolution` | `0.05` | Grid cell size in meters |
| `--max-step-height` | `0.15` | Max traversable height difference (m) |
| `--robot-height` | `0.5` | Vertical clearance required (m) |

### Download Example Data

```bash
huggingface-cli download pablovela5620/polycam-example-data 6G-room-example.zip --repo-type dataset --local-dir data/
```

### Platform Notes

- **aarch64 (Jetson/Grace)**: `open3d-unofficial-arm` is used automatically. PyTorch is sourced from the cu130 index.
- **x86_64**: Standard `open3d` and PyTorch CUDA wheels are used.

## Acknowledgements
Thanks to the original Prompt DepthAnything and DepthAnythingV2 repos!

[Prompt DepthAnything](https://github.com/DepthAnything/PromptDA)
```bibtex
@inproceedings{lin2024promptda,
  title={Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation},
  author={Lin, Haotong and Peng, Sida and Chen, Jingxiao and Peng, Songyou and Sun, Jiaming and Liu, Minghuan and Bao, Hujun and Feng, Jiashi and Zhou, Xiaowei and Kang, Bingyi},
  journal={arXiv},
  year={2024}
}
```

[DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)
```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
