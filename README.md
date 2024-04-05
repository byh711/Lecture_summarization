# **Leveraging Large Language Models (LLMs) for Lecture Summarization**


## **Introduction**

Access to educational resources plays a crucial role in the quality of learning Students often navigate through dense, information-rich lectures and face the pressure of mastering content swiftly to meet academic demands. The extensive amount of material, coupled with the limited time students have makes it challenging to distill the key pieces of information from lectures effectively. 

To that end, we’ve developed an approach to create comprehensive lecture summaries from lecture transcripts using Large Language Models (LLMs). We believe that by bridging the gap between long lectures and digestible written content, this project will help accelerate the learning process of students and significantly improve their ability to retain knowledge.

This project’s objective was to develop an LLM-based Lecture Note Generator that takes in lectures and creates comprehensive lecture summarises. This core component tool is envisioned to later incorporate STT technology and markdown templates to create an end-to-end pipeline that gives students access to high quality educational material that is relevant to their academic endeavours. With this tool in hand, students can spend more of their time paying attention and actively participating in classes as they rest assured that the essential knowledge will be captured in an easily accessible, succinct format for their later review.

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/cafddd9f-f5c7-42bf-9d9b-b3eb7136a4af" width="50%" height="50%">
</p>

### **Project Objectives**

- **To benchmark the performance of different LLMs** on lecture summarization.
- **To explore the effectiveness** of fine-tuning, zero-shot, and few-shot learning approaches.
- **To evaluate the potential of LLMs** to produce high-quality educational summaries that could aid in learning and knowledge retention.

## **Methodology**

### **Dataset**

The project utilizes the **CNN/DailyMail dataset** for pretraining and benchmarking, given its widespread use in text summarization tasks. This choice facilitates a focused evaluation of the LLMs' summarization capabilities on lecture content extracted from class recordings.

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/608cb060-4e3a-4f1d-bf9f-5133796bde68" width="600" height="400">
</p>

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/bf5884f9-d233-4c1d-876d-a657613d5529">
</p>

### **Making Datasets and DataLoaders**
```python
# Imports for handling datasets and dataframes
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd

# Load the CNN/Daily Mail dataset
train_data = load_dataset("cnn_dailymail", "3.0.0", split="train[:3000]")
val_data = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1000]")
test_data = load_dataset("cnn_dailymail", "3.0.0", split="test[:1000]")

# Function to preprocess the dataset
def preprocess_dataset(dataset):
    df = []
    for data in dataset:
        # Extract the article and highlights from the dataset
        question = data['article']
        answer = data['highlights']

        # Format the data as a prompt for the text summarization task
        prompt = f"<bos><start_of_turn>user\nYou are a helpful assistant for text summarization tasks. Once I provide you with the original content, please summarize it. Here is the content: {question}<end_of_turn>\n<start_of_turn>model\n{answer}<end_of_turn>"
        df.append(prompt)
    return df

# Preprocess and format the training dataset
train_dataset = preprocess_dataset(train_data)
train_dataset = pd.DataFrame({"text": train_dataset})
train_dataset = Dataset.from_pandas(train_dataset)

# Preprocess and format the validation dataset
val_dataset = preprocess_dataset(val_data)
val_dataset = pd.DataFrame({"text": val_dataset})
val_dataset = Dataset.from_pandas(val_dataset)

# Preprocess and format the test dataset
test_dataset = preprocess_dataset(test_data)
test_dataset = pd.DataFrame({"text": test_dataset})
test_dataset = Dataset.from_pandas(test_dataset)
```


## **Model Evaluated**

### **BART (406M)**

BART utilizes a Transformer-based neural machine translation architecture. It's designed with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT), making it highly effective for text summarization tasks.

### **GPT-2 (774M)**

GPT-2 employs a stacked Transformer decoder architecture for text generation, with a notable capacity for learning from diverse data, making it adept at generating coherent and contextually relevant summaries.

### **Llama 2 (7B)**

Both models are built on Transformer architectures, optimized for efficiency and scalability. Llama 2 and Gemma excel in understanding and generating human-like text, attributable to their extensive pre-training on vast datasets.

### **Gemma (2B)**

Both models are built on Transformer architectures, optimized for efficiency and scalability. Llama 2 and Gemma excel in understanding and generating human-like text, attributable to their extensive pre-training on vast datasets.

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/ebb527fd-fba4-4f2d-8fc3-66d3653a4bc1" width="800" height="500">
</p>


## **Performance Metrics Explained**

The evaluation of model performance utilized five key metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and BERT-Score. These metrics were chosen for their ability to capture different aspects of summarization quality:
- **ROUGE-N (where N=1, 2)** measures the overlap of N-grams between the generated summaries and reference texts, assessing the precision and recall of content.
- **ROUGE-L** focuses on the longest common subsequence, highlighting the fluency and sentence-level structure of the summaries.
- **BLEU** evaluates the precision of n-grams in the generated text compared to the reference, penalizing overly short or inaccurate outputs.
- **BERT-Score** leverages the contextual embeddings from BERT to assess semantic similarity between the generated and reference texts, providing a more nuanced understanding of summarization quality.

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/7824cf36-e141-4a25-93e8-cb62edc496d0" width="800" height="500">
</p>

## **Result**

The outcomes of our research are as follows:

- **Parameter Performance Correlation**: The results exhibit a clear correlation between the number of parameters in an LLM and its summarization performance. As illustrated in the accompanying chart, models with a higher number of parameters tend to produce more accurate and coherent summaries.

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/bc9deaa6-cb19-442d-a45b-5c20217588e6" width="600" height="400">
</p> 

- **Learning Approach Efficacy**: Particularly for the Llama2 and Gemma models, performances under zero-shot and few-shot learning conditions are comparable to those achieved with fine-tuning. This suggests that, at least for the lecture summarization task, satisfactory results can be obtained without the need for extensive fine-tuning, thanks to the pre-trained models' existing knowledge base.

- **Implications for Model Selection**: These findings have significant implications for the selection and deployment of LLMs in practical settings. Given the minimal performance difference, stakeholders can choose models like Gemma for their efficiency, even with fewer parameters, without sacrificing quality in the summarization task.


## Showcasing the Summarization Model

### **Course Lecture Summarization Example**

As a part of our project's application, we recorded a course lecture by Professor Sinisa Colic at the University of Toronto's MIE department. The lecture was transcribed using Google Cloud Platform's (GCP) Speech-to-Text API. The transcript was then processed through our best-performing model, Llama 2, to generate a summary.

#### **First Period (2024/03/19): Attention Mechanism**

During the first period, Professor Colic introduced the 'Attention Mechanism' in deep learning. Below is the workflow we followed for the summarization process:

1. **Recording**: Captured the lecture audio.
2. **Transcription**: Utilized GCP STT API for extracting accurate lecture transcripts.
3. **Summarization**: Input the transcript into the Llama 2 model to generate the summary.

**Transcript Example:**

> "One word at a time. And now this first hidden state or H not representation that meaning from French and then RNN decoder is going to try and output that in English. … All right so we'll take a break here for about 8 minutes or so. So we'll continue at."

**Generated Summary:**

> "Attention mechanisms and transformers are advancements in neural network architecture that improve the performance of neural machine translation systems, especially with longer sequences. … Transformers use self-attention to dynamically weigh the importance of different parts of the input data, enhancing the model's ability to handle long-range dependencies and complex patterns in data, and offering improved memory handling, interpretability, and flexibility in processing sequences."

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/348f742a-4490-45b5-8369-441839c1ef0e" width="800" height="500">
</p>

#### **Second Period (2024/03/19): Transformer Model**

In the second period, the topic was the 'Transformer' model. The same workflow was used as outlined above.

**Transcript Example:**

> "We addressed those largely with the attention mechanisms, Lstms. But yes, that issue grading is a limitation that we ran into but attention helps address that what's another limitation sequential their sequential complexity increases exactly. … Okay so let's, let's take a short break here and then."

**Generated Summary:**

> "Sequential processing in recurrent neural networks (RNNs) and Long Short-Term Memory (LSTM) networks can be slow and complex due to the need to process data sequentially, which limits parallel computation. … Transformers use self-attention mechanisms to process input data in parallel, enhancing efficiency and allowing the model to focus on different parts of the input simultaneously, overcoming the limitations of RNNs."

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/c670cf35-fb77-48a6-a18f-e4ae06e7a7a4" width="800" height="500">
</p>

The summaries produced by Llama 2 successfully condensed the essence of the lecture content, demonstrating the model's capability in an educational context.


## **Ablation Study**

An ablation study was conducted comparing the Fine-tuning, Zero-shot, and Few-shot performances of the Llama2 & Gemma models. Key findings include:
- **Minimal Performance Difference:** Indicating that both models possess inherent knowledge suitable for text summarization tasks, highlighting the potential redundancy of extensive fine-tuning for certain applications.
- **Importance of Prompt Engineering:** The study underscored how tailored prompts can significantly influence model output quality, suggesting that effective prompt design can yield desired outcomes even without specialized expertise.
- **Model Performance Comparison:** Despite Gemma's smaller size (2B parameters), it achieved a performance (BERT-Score of 42.3) close to that of Llama2 (7B parameters, BERT-Score of 45.3), demonstrating Gemma's efficiency.

<p align="center">
  <img src="https://github.com/byh711/Lecture_summarization/assets/63491899/bd3b4e60-ac66-4793-9c96-b32a6bb3d6c9" width="600" height="400">
</p> 

## **Detailed Case Study**

### **Pre-Processing Steps**

The lecture recording was first transcribed into text using an automated speech recognition (ASR) tool. This raw text was then segmented into logical sections corresponding to distinct topics within the lecture.

### **Model Application**

```python
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
lecture_text = "Here is the lecture text segmented into logical sections."
summary = summarizer(lecture_text, max_length=130, min_length=30, do_sample=False)
print(summary)
```

## **Summarization Examples**

**Before:** "The concept of neural networks is based on the understanding of biological neural networks."

**After:** "Neural networks mimic biological networks."

## **Discussion**

### **Limitations**

- **Resource Constraints**: The limited availability of GPU resources necessitated the use of LoRA quantization and smaller batch sizes, potentially impacting the models' performance.
- **Metric Limitations**: The chosen evaluation metrics were insufficient to capture certain nuances, such as awkward sentence structures or grammatical errors, underscoring the need for human evaluation.
- **Output Variability**: Inherent randomness in LLM outputs required the averaging of results across multiple trials to ascertain reliable performance benchmarks.

### **Implications**

The findings underscore **the significant potential of LLMs in educational applications**, particularly in automating the summarization of lecture content. Furthermore, the minimal performance difference between fine-tuning, zero-shot, and few-shot learning approaches opens up possibilities for deploying these models in resource-constrained environments.

## **Conclusion**

This project has demonstrated **the viability of using LLMs for the summarization of lecture content**, marking a promising step towards enhancing educational resources. The ability of models like Llama2 and Gemma to perform remarkably well, despite differences in their sizes, highlights the advancements in LLM technologies and their applicability in real-world scenarios. Moving forward, further research is needed to overcome the identified limitations and fully harness the capabilities of LLMs in education.

## **Future Directions**

### **Technological Advancements**

Future developments in LLMs could introduce models with even greater efficiency and nuanced understanding, further enhancing their summarization capabilities.

### **Educational Impact**

Widespread adoption of lecture summarization could revolutionize educational accessibility, engagement, and personalized learning, offering students tailored study materials and supporting diverse learning needs.

## **Ethical Considerations and Limitations**

### **Data Privacy**

Processing sensitive lecture content raises data privacy issues, necessitating robust safeguards to protect educational data.

### **Bias and Fairness**

LLMs may inherit biases from training data, potentially skewing summarizations. Addressing these biases is crucial for equitable educational resources.

### **Dependence on Technology**

While AI offers significant benefits, balancing technology with traditional educational methods is essential to prevent over-reliance and ensure comprehensive learning experiences.

## **Technical Appendix**

### **Code Snippets**

```python
# Sample code for applying Llama 2 model for summarization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("Llama-2")
tokenizer = AutoTokenizer.from_pretrained("Llama-2")
inputs = tokenizer.encode("Here is some input text.", return_tensors="pt")
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

## **Hyperparameter Settings**

For BART and Llama 2 models, the following hyperparameters were utilized:

- Learning rate: `2e-5`
- Batch size: `16`
- Number of epochs: `4`

## Related Works

In our final project, we address the problem of lecture summarization, a domain-specific extractive summarization from lecture scripts, which involves identifying the most important sentences in a script.

Many approaches to extractive summarization have been made since 2017, starting with *SummaRunner*, one of the early works using neural networks for text summarization. Since the emergence of Large Language Models (LLMs) in 2018, beginning with BERT, fine-tuning such models for downstream tasks has proven to be an effective and efficient method for achieving high-quality performance. Fine-tuning BERT for the lecture summarization task showcased BERT's capacity to select representative sentences from scripts, but challenges remained, such as identifying demonstrative pronouns (e.g., "these," "that").

We investigate the capacity of recent LLMs such as Llama 2 and Gemma, along with widely deployed models such as BART and GPT-2. BART combines bidirectional context in encoding with sequential decoding, demonstrating its capacity for accurate input data understanding and reconstruction, which is suitable for natural language generation tasks, including summarization. GPT-2, with nearly double the parameters of BART, excels at generating contextually relevant text, which is particularly useful for lecture summarization.

More recent works, Llama 2 by Meta AI and Gemma from Google, also demonstrate their ability to understand context accurately and generate coherent responses. Notably, Gemma, with 2 billion parameters, performs similarly to the 7 billion parameter model, Llama 2. The recent development of LLMs could mitigate the existing challenges of lecture summarization, particularly the understanding of pronouns from past sentences, and serve as a proxy to generate accurate and coherent summaries for lectures.

### Citations

- **BERT**: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. [arXiv preprint arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **Fine-tuning BERT**: Miller, D. (2019). Leveraging BERT for extractive text summarization on lectures. arXiv preprint arXiv:1906.04165. [arXiv preprint arXiv:1906.04165](https://arxiv.org/abs/1906.04165)
- **BART**: Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., ... & Zettlemoyer, L. (2019). Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461. [arXiv preprint arXiv:1910.13461](https://arxiv.org/abs/1910.13461)
- **GPT-2**: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9. [GPT-2_Radford_2018](https://ailab-ua.github.io/courses/resources/GPT-2_Radford_2018.pdf)
- **Llama 2**: Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., ... & Scialom, T. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. [arXiv preprint arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- **Gemma**: Team, G., Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., ... & Kenealy, K. (2024). Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295. [arXiv preprint arXiv:2403.08295](https://arxiv.org/abs/2403.08295)
- **Automatic Text Summarization Survey**: El-Kassas, W. S., Salama, C. R., Rafea, A. A., & Mohamed, H. K. (2021). Automatic text summarization: A comprehensive survey. Expert systems with applications, 165, 113679. [Expert Systems with Applications 165](https://www.sciencedirect.com/science/article/pii/S0957417420305030)
