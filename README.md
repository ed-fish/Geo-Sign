<div align="center">

<!-- logo / teaser image -->
<img src="https://github.com/user-attachments/assets/5e519f81-db4c-4f44-802f-bce208399d1c"
     alt="Geo-Sign logo" width="300"/>

<!-- title -->
<h1>
[NeurIPS 2025] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically-Aware Sign-Language Translation
</h1>

<!-- primary badges -->
<a href="https://arxiv.org/abs/2506.00129">
  <img src="https://img.shields.io/badge/arXiv-2506.00129-b31b1b.svg"/>
</a>
<a href="https://paperswithcode.com/sota/gloss-free-sign-language-translation-on-csl?p=geo-sign-hyperbolic-contrastive-1">
  <img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/geo-sign-hyperbolic-contrastive-1/gloss-free-sign-language-translation-on-csl"/>
</a>
<a href="https://huggingface.co/fiskenai/Geo-Sign">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-fiskenai/Geo--Sign-yellow"/>
</a>
<a href="https://opensource.org/licenses/MIT">
  <img src="https://img.shields.io/badge/License-MIT-green.svg"/>
</a>

<!-- line break between rows (optional) -->
<br/>

<!-- “tag” badges -->
<img src="https://img.shields.io/badge/Transformers-Library-blue"/>
<img src="https://img.shields.io/badge/CSL--Daily-Dataset-orange"/>
<img src="https://img.shields.io/badge/CSL--News-Dataset-orange"/>
<img src="https://img.shields.io/badge/Chinese-Language-red"/>
<img src="https://img.shields.io/badge/sign--language--translation-Task-success"/>
<img src="https://img.shields.io/badge/hyperbolic--geometry-Technique-purple"/>
</div>



This is the official implementation of the paper **"[Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation](https://arxiv.org/abs/2506.00129)"**.

Geo-Sign projects pose-based sign-language features into a learnable **Poincaré ball** and aligns them with text embeddings via a geometric contrastive loss. This approach improves translation performance by enhancing the model's understanding of the hierarchical relationships between different body parts (body, hands, face).

Compared with the strong Uni-Sign pose baseline, Geo-Sign boosts BLEU-4 by **+1.81** and ROUGE-L by **+3.03** on the CSL-Daily benchmark while keeping privacy-friendly skeletal inputs only.

This code is largely based on the Uni-Sign [ICLR 2025] project: https://github.com/ZechengLi19/Uni-Sign - we simply add hyperbolic regularisation on top of their method. If you use this code base you must also cite their work and give their repo a star. Thanks to the authors for help with the code setup and weights. 

<div align="center">
<img src="https://github.com/user-attachments/assets/55065fb2-ebe2-41ff-a374-1c5899aae8d3" width="600" />
</div>


---

## Key Features
* **Hyperbolic Contrastive Regularization**: Novel loss function in Poincaré space to learn geometrically structured representations.
* **ST-GCN Backbone**: Utilizes Spatio-Temporal Graph Convolutional Networks to effectively model pose dynamics.
* **mT5 Integration**: Leverages a pre-trained mT5 model for powerful sequence-to-sequence translation.
* **Distributed Training**: Full support for efficient, multi-GPU training using [DeepSpeed](https://github.com/microsoft/deepspeed).
* **Experiment Tracking**: Integrated with [Weights & Biases](https://wandb.ai) for easy logging and visualization.

  ## To-Do / Future Work
- [ ] Add support for additional sign language datasets (e.g How2Sign, YT-ASL).
- [ ] Explore different hyperbolic manifolds beyond the Poincaré ball.
- [ ] Develop more advanced visualization tools for geometric embeddings.
- [ ] Release a Gradio/Streamlit demo for interactive translation.
- [ ] Extend Hyperbolic embeddings to ST-GCN.
- [ ] Perform Hyperbolic pretraining on CSL News.

Want to help? Drop me an email and let's write a Geo-Sign++ paper! 

## Evaluation
| Method | Pose | RGB | Dev B-1 | Dev B-4 | Dev R-L | Test B-1 | Test B-4 | Test R-L |
| :--- | :--: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
| **_Gloss-Based Methods (Prior Art)_** | | | | | | | | |
| SLRT [6] | – | ✓ | 37.47 | 11.88 | 37.96 | 37.38 | 11.79 | 36.74 |
| TS-SLT [9] | ✓ | ✓ | 55.21 | 25.76 | 55.10 | 55.44 | 25.79 | 55.72 |
| CV-SLT [79] | – | ✓ | – | 28.24 | 56.36 | 58.29 | 28.94 | 57.06 |
| **_Gloss-Free Methods (Prior Art)_** | | | | | | | | |
| MSLU [83] | ✓ | – | 33.28 | 10.27 | 33.13 | 33.97 | 11.42 | 33.80 |
| SLRT [code](https://github.com/FangyunWei/SLRT) (Gloss-Free) | – | ✓ | 21.03 | 4.04 | 20.51 | 20.00 | 3.03 | 19.67 |
| GASLT [code](https://github.com/YinAoXiong/GASLT) | – | ✓ | – | – | – | 19.90 | 4.07 | 20.35 |
| GFSLT-VLP [code](https://github.com/zhoubenjia/GFSLT-VLP) | – | ✓ | 39.20 | 11.07 | 36.70 | 39.37 | 11.00 | 36.44 |
| FLa-LLM [paper](https://arxiv.org/abs/2403.12556) | – | ✓ | – | – | – | 37.13 | 14.20 | 37.25 |
| Sign2GPT [code](https://github.com/ryanwongsa/Sign2GPT) | – | ✓ | – | – | – | 41.75 | 15.40 | 42.36 |
| SignLLM [Project](https://signllm.github.io/) | – | ✓ | 42.45 | 12.23 | 39.18 | 39.55 | 15.75 | 39.91 |
| C2RL [paper](https://arxiv.org/html/2408.09949v1) | – | ✓ | – | – | – | 49.32 | 21.61 | 48.21 |
| **_Our Models and Baselines_** | | | | | | | | |
| Uni-Sign [code](https://github.com/ZechengLi19/Uni-Sign) (Pose) | ✓ | – | 53.24 | 25.27 | 54.34 | 53.86 | 25.61 | 54.92 |
| Uni-Sign [code](https://github.com/ZechengLi19/Uni-Sign) | ✓ | ✓ | 55.30 | 26.25 | 56.03 | 55.08 | 26.36 | 56.51 |
| Geo-Sign (Euclidean Pooled) | ✓ | – | 53.53 | 25.78 | 55.38 | 53.06 | 25.72 | 55.57 |
| Geo-Sign (Euclidean Token) | ✓ | – | 53.93 | 25.91 | 55.20 | 54.02 | 25.98 | 53.93 |
| Geo-Sign (Hyperbolic Pooled) | ✓ | – | 55.19 | 26.90 | 56.93 | 55.80 | 27.17 | 57.75 |
| **Geo-Sign (Hyperbolic Token)** | **✓** | **–** | **55.57** | **27.05** | **57.27** | **55.89** | **27.42** | **57.95** |


Geo-Sign outperforms all previous gloss-free pose-only methods and rivals many RGB- or gloss-based systems.

## Intended Uses & Scope
* **Primary Use**: Sign-language-to-text translation research, especially for resource-constrained or privacy-sensitive settings where RGB video is unavailable.
* **Out-of-scope**: Real-time production deployments without reliable pose estimation, medical or legal interpretations, or languages beyond the datasets the model was trained on.

---

## Setup and Installation

### 1. Clone the GitHub Repository
```bash
git clone https://github.com/ed-fish/geo-sign
cd geo-sign
```

### 2. Create Conda Environment
It is recommended to use Conda to create the environment from the provided `env.yaml` file for reproducibility.

```bash
conda env create -f env.yaml
conda activate Geo-Sign
```

### 3. Download Models & Data
You will need to download several files and place them in the correct directories within the cloned repository.

* **From the [fiskenai/Geo-Sign Hugging Face Repo](https://huggingface.co/fiskenai/Geo-Sign/tree/main):**
    * Download `pretraining.pth` -> Place in `./checkpoints/pretraining.pth`
    * Download `best.pth` -> Place in `./checkpoints/best.pth`
    * Download the entire `Data` folder -> Place at the root of the repo: `./Data/`
* **From [google/mt5-base on Hugging Face](https://huggingface.co/google/mt5-base):**
    * Download the model files and place them in `./pretrained_weight/mt5-base`.
* **From the [Uni-Sign Hugging Face Repo](https://huggingface.co/ZechengLi19/Uni-Sign/tree/main):**
  * Download the Poses for CSL Daily (and CSL News if you want to pretrain the hyperbolic model). Link to them in the `config.py` file

You can use the following Python snippet to download the mT5 model easily:
```python
from transformers import MT5ForConditionalGeneration, T5Tokenizer

model_name = "google/mt5-base"
save_directory = "./pretrained_weight/mt5-base"

MT5ForConditionalGeneration.from_pretrained(model_name).save_pretrained(save_directory)
T5Tokenizer.from_pretrained(model_name).save_pretrained(save_directory)
```

### 4. Configure Paths
All paths for datasets and pre-trained models are managed in `config.py`. Ensure the paths in this file correctly point to your local file structure, especially within the `./Data` directory you downloaded.

---
## Training

The main training script is `fine_tuning.py`. It uses `deepspeed` for distributed training.

### Single-GPU Training
Below is an example command for fine-tuning the model on a single GPU. The is the `script/train.sh` file.
```bash
# Set variables
output_dir=out/train
ckpt_path=checkpoints/pretraining.pth

deepspeed --num_gpus 1 fine_tuning.py \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  --epochs 40 \
  --opt AdamW \
  --lr 1e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task SLT \
  --use_hyperbolic \
  --wandb \
  --init_c 1.0 \
  --hyp_text_emb_src decoder \
  --hyp_text_cmp token \
  --alpha 0.7
```

To change from the `token` method to the `pooled` method as described in the paper - simply change the `--hyp_text_cmp_token` flag. 
To test Euclidean baselines you can either remove `--use_hyperbolic` which will remove all regularisation (i.e Uni-Sign Pose baseline), or set `--init_c` to 0.001 which will still perform regularisation but projections are basically euclidean.

### Multi-GPU Training
To train on multiple GPUs, simply change the `--num_gpus` argument in the `deepspeed` command to the number of GPUs you want to use. DeepSpeed will handle the rest.

For example, to train on 4 GPUs:
```bash
deepspeed --num_gpus 4 fine_tuning.py \
  --batch_size 8 \
  --gradient_accumulation_steps 8 \
  # ... (rest of the arguments remain the same)
```

## Evaluation
To run evaluation only, use the `--eval` flag. You must provide a path to a trained model checkpoint (e.g., the `best.pth` you downloaded) using the `--finetune` argument. You can check the `script/eval.sh` file.
```bash
# Set variables
output_dir=out/evaluation
ckpt_path=checkpoints/best.pth # Path to your best trained model

deepspeed --num_gpus 1 fine_tuning.py \
  --eval \
  --batch_size 8 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task SLT \
  --use_hyperbolic \
  --hyp_text_emb_src decoder \
  --hyp_text_cmp token
```

## Code Structure
```
.
├── fine_tuning.py      # Main script for training and evaluation
├── models.py           # Uni_Sign model architecture, including the hyperbolic branch
├── utils.py            # Helper functions, metric loggers, distributed setup, and argument parser
├── config.py           # Central configuration for all file paths
├── stgcn_layers.py     # Spatio-Temporal Graph Convolutional Network layers
├── datasets.py         # PyTorch dataset and dataloader logic
├── SLRT_metrics.py     # Evaluation metrics (BLEU, ROUGE, WER)
├── env.yaml            # Conda environment file
└── checkpoints/
    ├── best.pth
    └── pretraining.pth
└── pretrained_weight/
    └── mt5-base/
└── Data/
    └── ...
```

---
## Citation
If you use this work, please cite our paper:
```bibtex
@misc{fish2025geosign,
      title={Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation}, 
      author={Edward Fish and Richard Bowden},
      year={2025},
      eprint={2506.00129},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

The code and base architecture is directly taken from Uni-Sign. Thanks to the authors for their support with the code and weights. 
You must also cite their work if you use this method. 

```bibtex
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```

Please leave us a star if you appreciate the work :) 

