# KidsArtBench: Multi-Dimensional Children’s Art Evaluation with Attribute-Aware MLLMs

## Abstract
Multimodal Large Language Models (MLLMs) show remarkable progress across many visual–language tasks; however, their capacity to evaluate artistic expression remains limited: aesthetic concepts are inherently abstract and open-ended, and multimodal artwork annotations are scarce. We introduce KidsArtBench, a new benchmark of over 1k children's artworks (ages 5-15) annotated by 12 expert educators across 9 rubric-aligned dimensions, together with expert comments for feedback. Unlike prior aesthetic datasets that provide single scalar scores on adult imagery, KidsArtBench targets children's artwork and pairs multi-dimensional annotations with comment supervision to enable both ordinal assessment and formative feedback. Building on this resource, we propose an attribute-specific multi-LoRA approach -- where each attribute corresponds to a distinct evaluation dimension (e.g., Realism, Imagination) in the scoring rubric -- with Regression-Aware Fine-Tuning (RAFT) to align predictions with ordinal scales. On Qwen2.5-VL-7B, our method increases correlation from 0.468 to 0.653, with the largest gains on perceptual dimensions and narrowed gaps on higher-order attributes. These results show that educator-aligned supervision and attribute-aware training yield pedagogically meaningful evaluations and establish a rigorous testbed for sustained progress in educational AI. We release data and code with ethics documentation.

## Dataset
The annotation files of KidsArtBench are provided in the `ArtEduDataset/` directory of this repository. The corresponding artwork images were collected from children and are therefore required a signed Data Usage Agreement. Please complete and sign the agreement and send the signed copy to 51284118014@stu.ecnu.edu.cn or mingrui.ye@kcl.ac.uk, after which we will promptly provide access to the image data.

<img src="ArtEduDataset/fig1.png" width="600">

## Experiments
### Baseline Model
The baseline corresponds to the best-performing model reported in our paper. For reproducibility and ease of comparison, our model is released via the Hugging Face Hub: https://huggingface.co/BigRayss/KidArtBench
We provide multiple experiment scripts for different settings and ablation studies. 
Below we show the primary training and inference commands used in our main experiments.
### Train
> python experiment/train/qwen25_vl_7b_multi_lora_rail_sft.py
### Inference
> python experiment/evaluation/qwen25_vl_7b_multi_lora_rail_infer.py

## Reference

If you find **KidsArtBench** or the released baseline model useful for your research, please consider citing the following paper:

```bibtex
@article{ye2025kidsartbench,
  title={KidsArtBench: Multi-Dimensional Children's Art Evaluation with Attribute-Aware MLLMs},
  author={Ye, Mingrui and Zheng, Chanjin and Yu, Zengyi and Xiang, Chenyu and Zhao, Zhixue and Yuan, Zheng and Yannakoudakis, Helen},
  journal={arXiv preprint arXiv:2512.12503},
  year={2025}
}
