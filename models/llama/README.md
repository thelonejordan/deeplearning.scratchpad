# Llama 1

**Research Paper** ["LLaMA: Open and Efficient Foundation Language Models"](https://arxiv.org/pdf/2302.13971)

**Blog** ["Introducing LLaMA: A foundational, 65-billion-parameter large language model"](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)

## Model Index (Huggingface)

You can find all the original Llama checkpoints under the [Huggy Llama](https://huggingface.co/huggyllama) organization. (see https://huggingface.co/docs/transformers/en/model_doc/llama)

|Model|Llama1|
|---|---|
|7B| [Link](https://huggingface.co/huggyllama/llama-7b) |
|13B| [Link](https://huggingface.co/huggyllama/llama-13b) |
|33B| [Link](https://huggingface.co/huggyllama/llama-30b) |
|70B| [Link](https://huggingface.co/huggyllama/llama-65b) |

*The following info is copied from https://github.com/meta-llama/llama/blob/llama_v1/MODEL_CARD.md*

## Model details

**Organization developing the model**
The FAIR team of Meta AI.

**Model date**
LLaMA was trained between December. 2022 and Feb. 2023.

**Model version**
This is version 1 of the model.

**Model type**
LLaMA is an auto-regressive language model, based on the transformer architecture. The model comes in different sizes: 7B, 13B, 33B and 65B parameters.

**Paper or resources for more information**
More information can be found in the paper “LLaMA, Open and Efficient Foundation Language Models”, available at https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/.

## Factors

**Relevant factors**
One of the most relevant factors for which model performance may vary is which language is used. Although we included 20 languages in the training data, most of our dataset is made of English text, and we thus expect the model to perform better for English than other languages. Relatedly, it has been shown in previous studies that performance might vary for different dialects, and we expect that it will be the case for our model.

## Evaluation datasets
The model was evaluated on the following benchmarks: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC, OpenBookQA, NaturalQuestions, TriviaQA, RACE, MMLU, BIG-bench hard, GSM8k, RealToxicityPrompts, WinoGender, CrowS-Pairs.

## Training dataset
The model was trained using the following source of data: CCNet [67%], C4 [15%], GitHub [4.5%], Wikipedia [4.5%], Books [4.5%], ArXiv [2.5%], Stack Exchange[2%]. The Wikipedia and Books domains include data in the following languages: bg, ca, cs, da, de, en, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk. See the paper for more details about the training set and corresponding preprocessing.

## Quantitative analysis
Hyperparameters for the model architecture


<table>
    <thead>
            <tr>
            <th >LLaMA</th> <th colspan=6>Model hyper parameters </th>
            </tr>
            <tr>
            <th>Number of parameters</th><th>dimension</th><th>n heads</th><th>n layers</th><th>Learn rate</th><th>Batch size</th><th>n tokens</th>
            </tr>           
        </thead>
    <tbody>       
        <tr>
            <th>7B</th> <th>4096</th> <th>32</th> <th>32</th> <th>3.0E-04</th><th>4M</th><th>1T 
        </tr>
        <tr>
            <th>13B</th><th>5120</th><th>40</th><th>40</th><th>3.0E-04</th><th>4M</th><th>1T
        </tr>
        <tr>
            <th>33B</th><th>6656</th><th>52</th><th>60</th><th>1.5.E-04</th><th>4M</th><th>1.4T
        </tr>
        <tr>
            <th>65B</th><th>8192</th><th>64</th><th>80</th><th>1.5.E-04</th><th>4M</th><th>1.4T
        </tr>     
    </tbody>
</table>


*Table 1 - Summary of LLama Model Hyperparameters*
