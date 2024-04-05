# **Leveraging Large Language Models (LLMs) for Lecture Summarization**

## **Introduction**

In the realm of education, **distilling lectures into concise, informative summaries** remains a significant challenge. The advent of Large Language Models (LLMs) has opened new avenues for automating this task, potentially transforming how students engage with educational content. This project, conducted by Team 26, explores the application of LLMs to the task of lecture summarization, aiming to assess and compare the performance of various LLMs in this unique domain.

### **Project Objectives**

- **To benchmark the performance of different LLMs** on lecture summarization.
- **To explore the effectiveness** of fine-tuning, zero-shot, and few-shot learning approaches.
- **To evaluate the potential of LLMs** to produce high-quality educational summaries that could aid in learning and knowledge retention.

## **Methodology**

### **Dataset**

The project utilizes the **CNN/DailyMail dataset** for pretraining and benchmarking, given its widespread use in text summarization tasks. This choice facilitates a focused evaluation of the LLMs' summarization capabilities on lecture content extracted from class recordings.

![data_example](https://github.com/byh711/Lecture_summarization/assets/63491899/608cb060-4e3a-4f1d-bf9f-5133796bde68)

![data_distribution](https://github.com/byh711/Lecture_summarization/assets/63491899/bf5884f9-d233-4c1d-876d-a657613d5529)

<img src = "[Your Image Addr](https://github.com/byh711/Lecture_summarization/assets/63491899/608cb060-4e3a-4f1d-bf9f-5133796bde68)" width="50%" height="50%">


## **Model Evaluated**

### **BART (406M)**

BART utilizes a Transformer-based neural machine translation architecture. It's designed with a bidirectional encoder (like BERT) and a left-to-right decoder (like GPT), making it highly effective for text summarization tasks.

### **GPT-2 (774M)**

GPT-2 employs a stacked Transformer decoder architecture for text generation, with a notable capacity for learning from diverse data, making it adept at generating coherent and contextually relevant summaries.

### **Llama 2 (7B)**

Both models are built on Transformer architectures, optimized for efficiency and scalability. Llama 2 and Gemma excel in understanding and generating human-like text, attributable to their extensive pre-training on vast datasets.

### **Gemma (2B)**

Both models are built on Transformer architectures, optimized for efficiency and scalability. Llama 2 and Gemma excel in understanding and generating human-like text, attributable to their extensive pre-training on vast datasets.

![Model](https://github.com/byh711/Lecture_summarization/assets/63491899/ebb527fd-fba4-4f2d-8fc3-66d3653a4bc1)

### **Performance Metrics Explained**

The evaluation of model performance utilized five key metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, and BERT-Score. These metrics were chosen for their ability to capture different aspects of summarization quality:
- **ROUGE-N (where N=1,2)** measures the overlap of N-grams between the generated summaries and reference texts, assessing the precision and recall of content.
- **ROUGE-L** focuses on the longest common subsequence, highlighting the fluency and sentence-level structure of the summaries.
- **BLEU** evaluates the precision of n-grams in the generated text compared to the reference, penalizing overly short or inaccurate outputs.
- **BERT-Score** leverages the contextual embeddings from BERT to assess semantic similarity between the generated and reference texts, providing a more nuanced understanding of summarization quality.

![Metric](https://github.com/byh711/Lecture_summarization/assets/63491899/7824cf36-e141-4a25-93e8-cb62edc496d0)

### **Result**

The outcomes of our research are as follows:

- **Parameter Performance Correlation**: The results exhibit a clear correlation between the number of parameters in an LLM and its summarization performance. As illustrated in the accompanying chart, models with a higher number of parameters tend to produce more accurate and coherent summaries.

  ![Result](https://github.com/byh711/Lecture_summarization/assets/63491899/bc9deaa6-cb19-442d-a45b-5c20217588e6)

- **Learning Approach Efficacy**: Particularly for the Llama2 and Gemma models, performances under zero-shot and few-shot learning conditions are comparable to those achieved with fine-tuning. This suggests that, at least for the lecture summarization task, satisfactory results can be obtained without the need for extensive fine-tuning, thanks to the pre-trained models' existing knowledge base.

- **Implications for Model Selection**: These findings have significant implications for the selection and deployment of LLMs in practical settings. Given the minimal performance difference, stakeholders can choose models like Gemma for their efficiency, even with fewer parameters, without sacrificing quality in the summarization task.

### **Ablation Study**

An ablation study was conducted comparing the Fine-tuning, Zero-shot, and Few-shot performances of the Llama2 & Gemma models. Key findings include:
- **Minimal Performance Difference:** Indicating that both models possess inherent knowledge suitable for text summarization tasks, highlighting the potential redundancy of extensive fine-tuning for certain applications.
- **Importance of Prompt Engineering:** The study underscored how tailored prompts can significantly influence model output quality, suggesting that effective prompt design can yield desired outcomes even without specialized expertise.
- **Model Performance Comparison:** Despite Gemma's smaller size (2B parameters), it achieved a performance (BERT-Score of 42.3) close to that of Llama2 (7B parameters, BERT-Score of 45.3), demonstrating Gemma's efficiency.

  ![Ablation](https://github.com/byh711/Lecture_summarization/assets/63491899/bd3b4e60-ac66-4793-9c96-b32a6bb3d6c9)

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
