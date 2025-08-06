# Dynamic Phase-aware Hallucination Correction with Iteratively Focused Intervention for Large Vision-Language Models

## Project Overview

Large Vision-Language Models (LVLMs) have revolutionized cross-modal understanding, but still struggle with vision hallucination, where the generated content is inconsistent with the visual input.
Existing hallucination mitigation approaches either rely on prohibitive computational costs to fine-tune LVLMs or apply fixed post-hoc intervention strategies, which overlooks the dynamic patterns of hallucination emergence. 
In this paper, we identify that vision hallucination exhibits dynamic phase-wise patterns: Hallucination severity varies across different phases of generation, but tends to be most pronounced at the start of each phase.
Building upon this, we propose FocalPhase, a dynamic hallucination mitigation framework that performs iterative and phase-aware correction with focused intervention, eliminating the need for additional data or fine-tuning LVLMs.
Specifically, we leverage hallucination discrimination capacity of LVLMs to train a lightweight sentence-level hallucination detector via confidence-aware contrastive learning.  
At inference time, the detector enables real-time hallucination detection to guide LVLMs to perform iteratively focused correction  at the early decoding stages within each phase, until hallucinations are resolved.
Extensive experiments on hallucination benchmarks show that FocalPhase outperforms the previous post-hoc baselines and approach the methods of  fine-tuning LVLMs.
Further analysis reveals that our method achieves strong performance within a controllable iterative steps, offering a favorable balance between accuracy and inference efficiency.
