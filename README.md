# Prompt Depth Anything Example

## Installation
### Using Pixi
Make sure you have the [Pixi](https://pixi.sh/latest/#installation) package manager installed
```bash
git clone https://github.com/rerun-io/prompt-da.git
cd prompt-da
pixi run app
```

All commands can be listed using `pixi task list`

## Usage
### Gradio App
```
pixi run app
```
### CLI
with pixi example task
```bash
pixi run polycam-prompt_da
```

with python in pixi shell
```bash
python tools/prompt_da_polycam.py --polycam-zip-path $PATH_TO_POLYCAM_ZIP
```

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