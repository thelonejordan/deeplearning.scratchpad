# [Llama 2](https://www.llama.com/llama2/)

**Blog** ["Meta and Microsoft Introduce the Next Generation of Llama"](https://ai.meta.com/blog/llama-2/)



*The following info is copied from http://huggingface.co/meta-llama/Llama-2-7b/blob/main/README.md, which is an extension of https://github.com/meta-llama/llama/blob/llama_v2/MODEL_CARD.md*

## Model Details
*Note: Use of this model is governed by the Meta license. In order to download the model weights and tokenizer, please visit the [website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept our License before requesting access here.*

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama-2-Chat, are optimized for dialogue use cases. Llama-2-Chat models outperform open-source chat models on most benchmarks we tested, and in our human evaluations for helpfulness and safety, are on par with some popular closed-source models like ChatGPT and PaLM.

**Model Developers** Meta

**Variations** Llama 2 comes in a range of parameter sizes — 7B, 13B, and 70B — as well as pretrained and fine-tuned variations.

**Input** Models input text only.

**Output** Models generate text only.

**Model Architecture** Llama 2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align to human preferences for helpfulness and safety.


||Training Data|Params|Content Length|GQA|Tokens|LR|
|---|---|---|---|---|---|---|
|Llama 2|*A new mix of publicly available online data*|7B|4k|&#10007;|2.0T|3.0 x 10<sup>-4</sup>|
|Llama 2|*A new mix of publicly available online data*|13B|4k|&#10007;|2.0T|3.0 x 10<sup>-4</sup>|
|Llama 2|*A new mix of publicly available online data*|70B|4k|&#10004;|2.0T|1.5 x 10<sup>-4</sup>|

*Llama 2 family of models.* Token counts refer to pretraining data only. All models are trained with a global batch-size of 4M tokens. Bigger models -  70B -- use Grouped-Query Attention (GQA) for improved inference scalability.

**Model Dates** Llama 2 was trained between January 2023 and July 2023.

**Research Paper** ["Llama-2: Open Foundation and Fine-tuned Chat Models"](arxiv.org/abs/2307.09288)

## Intended Use

**Intended Use Cases** Llama 2 is intended for commercial and research use in English. Tuned models are intended for assistant-like chat, whereas pretrained models can be adapted for a variety of natural language generation tasks.

To get the expected features and performance for the chat versions, a specific formatting needs to be followed, including the `INST` and `<<SYS>>` tags, `BOS` and `EOS` tokens, and the whitespaces and breaklines in between (we recommend calling `strip()` on inputs to avoid double-spaces). See our reference code in github for details: [`chat_completion`](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212).

## Training Data

**Overview** Llama 2 was pretrained on 2 trillion tokens of data from publicly available sources. The fine-tuning data includes publicly available instruction datasets, as well as over one million new human-annotated examples. Neither the pretraining nor the fine-tuning datasets include Meta user data.

**Data Freshness** The pretraining data has a cutoff of September 2022, but some tuning data is more recent, up to July 2023.

## Model Index (Huggingface)

|Model|Llama2|Llama2-hf|Llama2-chat|Llama2-chat-hf|
|---|---|---|---|---|
|7B| [Link](https://huggingface.co/meta-llama/Llama-2-7b) | [Link](https://huggingface.co/meta-llama/Llama-2-7b-hf) | [Link](https://huggingface.co/meta-llama/Llama-2-7b-chat) | [Link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)|
|13B| [Link](https://huggingface.co/meta-llama/Llama-2-13b) | [Link](https://huggingface.co/meta-llama/Llama-2-13b-hf) | [Link](https://huggingface.co/meta-llama/Llama-2-13b-chat) | [Link](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)|
|70B| [Link](https://huggingface.co/meta-llama/Llama-2-70b) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-hf) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-chat) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)|
