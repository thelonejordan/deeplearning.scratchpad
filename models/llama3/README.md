# [Llama 3](https://www.llama.com/models/llama-3/)

**Research Paper** ["The Llama 3 Herd of Models"](https://arxiv.org/pdf/2407.21783)

**Blog** ["Introducing Llama 3.1: Our most capable models to date"](https://ai.meta.com/blog/meta-llama-3-1/)

**Blog** ["Introducing Meta Llama 3: The most capable openly available LLM to date"](https://ai.meta.com/blog/meta-llama-3/)

**Blog** ["Llama 3.2: Revolutionizing edge AI and vision with open, customizable models"](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

**Model Card** ["Llama 3.1 - Model Cards & Prompt formats"](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)

**Model Card** ["Llama 3.2 - Model Cards & Prompt formats"](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/)

**Model Card** ["Llama 3.3 - Model Cards & Prompt formats"](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/)

## Model Index (Huggingface)

[Llama3 Collection](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)

[Llama3.1 Collection](https://huggingface.co/collections/meta-llama/llama-31-669fc079a0c406a149a5738f)

[Llama3.2 Collection](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)

[LLama3.3 Collection](https://huggingface.co/collections/meta-llama/llama-33-67531d5c405ec5d08a852000)

---

*The following info is copied from https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/README.md, which is an extension of https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md*

## Model Details

Meta developed and released the Meta Llama 3 family of large language models (LLMs), a collection of pretrained and instruction tuned generative text models in 8 and 70B sizes. The Llama 3 instruction tuned models are optimized for dialogue use cases and outperform many of the available open source chat models on common industry benchmarks. Further, in developing these models, we took great care to optimize helpfulness and safety. 

**Model developers** Meta

**Variations** Llama 3 comes in two sizes — 8B and 70B parameters — in pre-trained and instruction tuned variants.

**Input** Models input text only.

**Output** Models generate text and code only.

**Model Architecture** Llama 3 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.


<table>
  <tr>
   <td>
   </td>
   <td><strong>Training Data</strong>
   </td>
   <td><strong>Params</strong>
   </td>
   <td><strong>Context length</strong>
   </td>
   <td><strong>GQA</strong>
   </td>
   <td><strong>Token count</strong>
   </td>
   <td><strong>Knowledge cutoff</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" >Llama 3
   </td>
   <td rowspan="2" >A new mix of publicly available online data.
   </td>
   <td>8B
   </td>
   <td>8k
   </td>
   <td>Yes
   </td>
   <td rowspan="2" >15T+
   </td>
   <td>March, 2023
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>8k
   </td>
   <td>Yes
   </td>
   <td>December, 2023
   </td>
  </tr>
</table>


**Llama 3 family of models**. Token counts refer to pretraining data only. Both the 8 and 70B versions use Grouped-Query Attention (GQA) for improved inference scalability.

**Model Release Date** April 18, 2024.


## Training Data

**Overview** Llama 3 was pretrained on over 15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 10M human-annotated examples. Neither the pretraining nor the fine-tuning datasets include Meta user data.

**Data Freshness** The pretraining data has a cutoff of March 2023 for the 8B and December 2023 for the 70B models respectively. 


---

*The following info is copied from https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/README.md, which is an extension of https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md*

## Model Information

The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction tuned generative models in 8B, 70B and 405B sizes (text in/text out). The Llama 3.1 instruction tuned text only models (8B, 70B, 405B) are optimized for multilingual dialogue use cases and outperform many of the available open source and closed chat models on common industry benchmarks.

**Model developer**: Meta

**Model Architecture:** Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. 


<table>
  <tr>
   <td>
   </td>
   <td><strong>Training Data</strong>
   </td>
   <td><strong>Params</strong>
   </td>
   <td><strong>Input modalities</strong>
   </td>
   <td><strong>Output modalities</strong>
   </td>
   <td><strong>Context length</strong>
   </td>
   <td><strong>GQA</strong>
   </td>
   <td><strong>Token count</strong>
   </td>
   <td><strong>Knowledge cutoff</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" >Llama 3.1 (text only)
   </td>
   <td rowspan="3" >A new mix of publicly available online data.
   </td>
   <td>8B
   </td>
   <td>Multilingual Text
   </td>
   <td>Multilingual Text and code
   </td>
   <td>128k
   </td>
   <td>Yes
   </td>
   <td rowspan="3" >15T+
   </td>
   <td rowspan="3" >December 2023
   </td>
  </tr>
  <tr>
   <td>70B
   </td>
   <td>Multilingual Text
   </td>
   <td>Multilingual Text and code
   </td>
   <td>128k
   </td>
   <td>Yes
   </td>
  </tr>
  <tr>
   <td>405B
   </td>
   <td>Multilingual Text
   </td>
   <td>Multilingual Text and code
   </td>
   <td>128k
   </td>
   <td>Yes
   </td>
  </tr>
</table>


**Supported languages:** English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

**Llama 3.1 family of models**. Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability.

**Model Release Date:** July 23, 2024.

## Training Data

**Overview:** Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 25M synthetically generated examples. 

**Data Freshness:** The pretraining data has a cutoff of December 2023.


---

*The following info is copied from https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/README.md, which is an extension of https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md*

## Model Information

The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out). The Llama 3.2 instruction-tuned text only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. They outperform many of the available open source and closed chat models on common industry benchmarks.

**Model Developer:** Meta

**Model Architecture:** Llama 3.2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

|  | Training Data | Params | Input modalities | Output modalities | Context Length | GQA | Shared Embeddings | Token count | Knowledge cutoff |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Llama 3.2 (text only)  | A new mix of publicly available online data. | 1B (1.23B) | Multilingual Text | Multilingual Text and code  | 128k | Yes | Yes | Up to 9T tokens | December 2023 |
|  |  | 3B (3.21B) | Multilingual Text | Multilingual Text and code  |  |  |  |  |  |
| Llama 3.2 Quantized (text only)  | A new mix of publicly available online data. | 1B (1.23B) | Multilingual Text | Multilingual Text and code  | 8k | Yes | Yes | Up to 9T tokens | December 2023 |
|  |  | 3B (3.21B) | Multilingual Text | Multilingual Text and code |  |  |  |  |  |

**Supported Languages:** English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported. Llama 3.2 has been trained on a broader collection of languages than these 8 supported languages. Developers may fine-tune Llama 3.2 models for languages beyond these supported languages, provided they comply with the Llama 3.2 Community License and the Acceptable Use Policy. Developers are always expected to ensure that their deployments, including those that involve additional languages, are completed safely and responsibly.

**Llama 3.2 Model Family:** Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability.

**Model Release Date:** Sept 25, 2024

## Training Data

**Overview:** Llama 3.2 was pretrained on up to 9 trillion tokens of data from publicly available sources. For the 1B and 3B Llama 3.2 models, we incorporated logits from the Llama 3.1 8B and 70B models into the pretraining stage of the model development, where outputs (logits) from these larger models were used as token-level targets. Knowledge distillation was used after pruning to recover performance. In post-training we used a similar recipe as Llama 3.1 and produced final chat models by doing several rounds of alignment on top of the pre-trained model. Each round involved Supervised Fine-Tuning (SFT), Rejection Sampling (RS), and Direct Preference Optimization (DPO).

**Data Freshness:** The pretraining data has a cutoff of December 2023\.


---

*The following info is copied from https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/README.md, which is an extension of https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md*

## Model Information

The Meta Llama 3.3 multilingual large language model (LLM) is an instruction tuned generative model in 70B (text in/text out). The Llama 3.3 instruction tuned text only model is optimized for multilingual dialogue use cases and outperforms many of the available open source and closed chat models on common industry benchmarks.

**Model developer**: Meta

**Model Architecture:** Llama 3.3 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety. 

|  | Training Data | Params | Input modalities | Output modalities | Context length | GQA | Token count | Knowledge cutoff |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Llama 3.3 (text only)  | A new mix of publicly available online data. | 70B | Multilingual Text | Multilingual Text and code  | 128k | Yes | 15T+ | December 2023 |

**Supported languages:** English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.

**Llama 3.3 model**. Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability.

**Model Release Date:** 

* **70B Instruct: December 6, 2024**

## Training Data

**Overview:** Llama 3.3 was pretrained on \~15 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over 25M synthetically generated examples. 

**Data Freshness:** The pretraining data has a cutoff of December 2023\.
