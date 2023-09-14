<div align="center">
  <img src="./imgs/log-llmenhance.png" alt="LLM enhancement" width="500"><br>
  <strong>
    Original research on enhancement of LLMs conducted by Microsoft Research and other collaborated institutes.
  </strong><br>
  (Contact: <a href="https://jd92.wang/">Jindong Wang</a>, also see our projects on <a href="https://llm-eval.github.io/">LLM evaluation</a>)
</div>
<br>

- [1. EmotionPrompt: Leveraging Psychology for Enhancement via Emotional Stimulus](#1-emotionprompt-leveraging-psychology-for-enhancement-via-emotional-stimulus)
- [2. Exploring Vision-Language Models for Imbalanced Learning](#2-exploring-vision-language-models-for-imbalanced-learning)
- [3. FedCLIP: Fast Generalization and Personalization for CLIP in Federated Learning](#3-fedclip-fast-generalization-and-personalization-for-clip-in-federated-learning)

- - -

## 1. EmotionPrompt: Leveraging Psychology for Enhancement via Emotional Stimulus
 
<p align="center">
  Cheng Li<sup>1</sup>,
  Jindong Wang<sup>#2</sup>, 
  Kaijie Zhu<sup>2</sup>, 
  Yixuan Zhang<sup>3</sup>,  
  Wenxin Hou<sup>2</sup>, 
  Jianxun Lian<sup>2</sup>, 
  Xing Xie<sup>2</sup>
</p>  

<p align="center">
<sup>1</sup> Institute of Software, CAS,
<sup>2</sup> Microsoft,
<sup>3</sup> College of William and Mary<br>
(#: Co-corresponding author)
</p>

<p align="center">
[<a href="https://arxiv.org/pdf/2307.11760.pdf">Paper</a>]
[Media coverage <a href="https://techxplore.com/news/2023-08-exploring-effects-emotional-stimuli-large.html">1</a>,<a href="https://webappia.com/investigating-the-impact-of-emotional-stimuli-on-language-models-emotionalai/?feed_id=13302&_unique_id=64cfac8201058">2</a>,<a href="https://flowgpt.com/blog/emoGPT">3</a>,<a href="https://mp.weixin.qq.com/s/z-lg-yDibKoT_gO97HSK5Q">4</a>]
</p>

<details>
<summary>Abstract</summary>
Large language models (LLMs) have achieved significant performance in many fields such as reasoning, language understanding, and math problem-solving, and are regarded as a crucial step to artificial general intelligence (AGI). However, the sensitivity of LLMs to prompts remains a major bottleneck for their daily adoption. In this paper, we take inspiration from psychology and propose EmotionPrompt to explore emotional intelligence to enhance the performance of LLMs. EmotionPrompt operates on a remarkably straightforward principle: the incorporation of emotional stimulus into prompts. Experimental results demonstrate that our EmotionPrompt, using the same single prompt templates, significantly outperforms original zero-shot prompt and Zero-shot-CoT on 8 tasks with diverse models: ChatGPT, Vicuna-13b, Bloom, and T5. Further, EmotionPrompt was observed to improve both truthfulness and informativeness. We believe that EmotionPrompt heralds a novel avenue for exploring interdisciplinary knowledge for humans-LLMs interaction.
</details>

<p align="center">
<img src="./imgs/emotion.png" style="width: 70%;"/>
</p>

- - -

## 2. Exploring Vision-Language Models for Imbalanced Learning

*This work accepted minor revision at **IJCV 2023**.*

<p align="center">
  Yidong Wang<sup>1</sup>,
  Zhuohao Yu<sup>1</sup>,
  Jindong Wang<sup>#2</sup>,  
  Qiang Heng<sup>3</sup>,  
  Hao Chen<sup>4</sup>, 
  Wei Ye<sup>#1</sup>,  
  Rui Xie<sup>1</sup>, 
  Xing Xie<sup>2</sup>, 
  Shikun Zhang<sup>#1</sup><br>
</p>  

<p align="center">
<sup>1</sup> Peking University
<sup>2</sup> Microsoft Research,
<sup>3</sup> North Carolina State University,
<sup>4</sup> Carnegie Mellon University<br>
(#: Corresponding authors)
</p>

<p align="center">
[<a href="https://arxiv.org/pdf/2304.01457.pdf">Paper</a>]
[<a href="https://github.com/Imbalance-VLM/Imbalance-VLM">Github</a>]
</p>

<details>
<summary>Abstract</summary>
Vision-Language models (VLMs) that use contrastive language-image pre-training have shown promising zero-shot classification performance. However, their performance on imbalanced dataset is relatively poor, where the distribution of classes in the training dataset is skewed, leading to poor performance in predicting minority classes. For instance, CLIP achieved only 5% accuracy on the iNaturalist18 dataset. We propose to add a lightweight decoder to VLMs to avoid OOM (out of memory) problem caused by large number of classes and capture nuanced features for tail classes. Then, we explore improvements of VLMs using prompt tuning, fine-tuning, and incorporating imbalanced algorithms such as Focal Loss, Balanced SoftMax and Distribution Alignment. Experiments demonstrate that the performance of VLMs can be further boosted when used with decoder and imbalanced methods. Specifically, our improved VLMs significantly outperforms zero-shot classification by an average accuracy of 6.58%, 69.82%, and 6.17%, on ImageNet-LT, iNaturalist18, and Places-LT, respectively. We further analyze the influence of pre-training data size, backbones, and training cost. Our study highlights the significance of imbalanced learning algorithms in face of VLMs pre-trained by huge data. We release our code at https://github.com/Imbalance-VLM/Imbalance-VLM.

</details>


<p align="center">
<img src="./imgs/imbalance.png" style="width: 70%;"/>
</p>

- - -

## 3. FedCLIP: Fast Generalization and Personalization for CLIP in Federated Learning

*This work is published at **IEEE Data Engineering Bulletin** 2023.*

<p align="center">
  Wang Lu<sup>1</sup>,  
  Xixu Hu<sup>2</sup>,  
  Jindong Wang<sup>#3</sup>,  
  Xing Xie<sup>3</sup><br>
  (#: Corresponding authors)
</p>  

<p align="center">
<sup>1</sup> Chinese Academy of Sciences
<sup>2</sup> City University of Hong Kong,
<sup>3</sup> Microsoft Research<br>
</p>

<p align="center">
[<a href="https://arxiv.org/pdf/2302.13485v2.pdf">Paper</a>]
[<a href="https://github.com/microsoft/PersonalizedFL">Github</a>]
</p>

<details>
<summary>Abstract</summary>
Federated learning (FL) has emerged as a new paradigm for privacy-preserving computation in recent years. Unfortunately, FL faces two critical challenges that hinder its actual performance: data distribution heterogeneity and high resource costs brought by large foundation models. Specifically, the non-IID data in different clients make existing FL algorithms hard to converge while the high resource costs, including computational and communication costs that increase the deployment difficulty in real-world scenarios. In this paper, we propose an effective yet simple method, named FedCLIP, to achieve fast generalization and personalization for CLIP in federated learning. Concretely, we design an attention-based adapter for the large model, CLIP, and the rest operations merely depend on adapters. Lightweight adapters can make the most use of pretrained model information and ensure models be adaptive for clients in specific tasks. Simultaneously, small-scale operations can mitigate the computational burden and communication burden caused by large models. Extensive experiments are conducted on three datasets with distribution shifts. Qualitative and quantitative results demonstrate that FedCLIP significantly outperforms other baselines (9% overall improvements on PACS) and effectively reduces computational and communication costs (283x faster than FedAVG). Our code will be available at: https://github.com/microsoft/PersonalizedFL.

</details>

<p align="center">
<img src="./imgs/fedclip.png" style="width: 50%;"/>
</p>

