<div align="center">

# Dynamic Phase-aware Hallucination Correction with Iteratively Focused Intervention for Large Vision-Language Models

</div>

## Project Overview

Large Vision-Language Models (LVLMs) have revolutionized cross-modal understanding, but still struggle with vision hallucination, where the generated content is inconsistent with the visual input.
Existing hallucination mitigation approaches either rely on prohibitive computational costs to fine-tune LVLMs or apply fixed post-hoc intervention strategies, which overlooks the dynamic patterns of hallucination emergence. 
In this paper, we identify that vision hallucination exhibits dynamic phase-wise patterns: Hallucination severity varies across different phases of generation, but tends to be most pronounced at the start of each phase.
Building upon this, we propose FocalPhase, a dynamic hallucination mitigation framework that performs iterative and phase-aware correction with focused intervention, eliminating the need for additional data or fine-tuning LVLMs.
Specifically, we leverage hallucination discrimination capacity of LVLMs to train a lightweight sentence-level hallucination detector via confidence-aware contrastive learning.  
At inference time, the detector enables real-time hallucination detection to guide LVLMs to perform iteratively focused correction  at the early decoding stages within each phase, until hallucinations are resolved.
Extensive experiments on hallucination benchmarks show that FocalPhase outperforms the previous post-hoc baselines and approach the methods of  fine-tuning LVLMs.
Further analysis reveals that our method achieves strong performance within a controllable iterative steps, offering a favorable balance between accuracy and inference efficiency.

## Install 🛠️
First install the packages required for our project.
```bash
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
pip install en_core_web_lg-3.8.0-py3-none-any.whl
```
Then download COCO train2014 images and unzip it as data/train2014.
```bash
cd data
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
```

## Data
###  Vision hallucination detector training dataset
We provide the scrips of generating vision hallucination detection training data with LLaVA-1.5.
[scrips](#section1)

### Our hallucination correction result
We provide the AMBER generative result of LLaVA-1.5-7B with our FocalPhase Hallucination Correction method.
[AMBER generative result](#section2)

## Code
### Generate vision hallucination detection training data
<a id="section1"></a>
```bash
python generate_coco_list.py
python model_vqa_batch_noise_release.py
bash llava_caption_sentence_judge_stage1.sh
bash llava_caption_sentence_judge_stage2.sh
```

### Fine-tune CLIP for vision hallucination detection
The training code will be made publicly available upon acceptance of the paper.

### Post-hoc hallucination correction for LVLMs
The correction code will be made publicly available upon acceptance of the paper.

### Evaluate hallucination correction results for LVLMs
<a id="section2"></a>
```bash
python eval/amber/inference.py  --inference_data exp/gen_llava157_amber_focalphase.jsonl
```
