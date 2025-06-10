# Fine-Tuning DistilBERT on GoEmotions for Sentiment Analysis

This project demonstrates how to fine-tune the DistilBERT model on the GoEmotions dataset to perform sentiment analysis. The notebook provides a step-by-step guide, from data preprocessing to model evaluation.
#### [Link To Fine Tuned Model](https://drive.google.com/drive/folders/1gmA4-s1gCMfC9k4XUWVKyo7ixeBI-36V?usp=sharing)
#### [Link to colab nb](https://colab.research.google.com/github/kanchanraiii/SentimentAnalysis_GoEmotions/blob/main/GoEmotions_FineTuneDistilBERT.ipynb)
## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [References](#references)

## Introduction

Sentiment analysis is a crucial task in natural language processing (NLP) that involves determining the emotional tone behind textual data. This project utilizes the GoEmotions dataset, a comprehensive collection of human-annotated Reddit comments categorized into 27 emotion labels, to fine-tune DistilBERT—a smaller, faster, and lighter version of BERT.

## Dataset

The [GoEmotions dataset](https://github.com/google-research/google-research/tree/master/goemotions) consists of approximately 58,000 Reddit comments labeled across 27 emotion categories, including happiness, sadness, anger, and more. This rich dataset enables the development of models capable of nuanced emotion detection.

## Model

[DistilBERT](https://arxiv.org/abs/1910.01108) is a distilled version of BERT, retaining 97% of its language understanding while being 60% faster and lighter. Fine-tuning DistilBERT on the GoEmotions dataset allows for efficient and effective sentiment analysis.

## Dependencies

To run the notebook, ensure you have the following dependencies installed:

- Python 3.x
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

You can install the required packages using:

```bash
pip install transformers datasets torch scikit-learn pandas numpy
```
## References
- GoEmotions Dataset: https://github.com/google-research/google-research/tree/master/goemotions
- DistilBERT Paper: https://arxiv.org/abs/1910.01108
- Hugging Face Transformers: https://github.com/huggingface/transformers

