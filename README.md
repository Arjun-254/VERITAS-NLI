# VERITAS-NLI : Validation and Extraction of Reliable Information Through Automated Scraping and Natural Language Inference
This repository contains the artifacts for our paper titled `VERITAS-NLI : Validation and Extraction of Reliable Information Through Automated Scraping and Natural Language Inference` 

Our Proposed solution utilizes `Real-Time Web Scraping` and `Natural Language Inference(NLI)` Models for the task of Fake News Detection :
- Dynamic web scraping allows for real-time external knowledge retrieval for the fact-checking of headlines.
- Natural Language Inference models are used to find support for the claimed headline in the web scraped text, to ascertain the veracity of an input headline

## Visual Abstract

<img width="1455" alt="Screenshot 2024-12-02 at 4 52 44 PM" src="https://github.com/user-attachments/assets/9c142709-cb4d-4338-934d-dee0ccc8944b">

## Directory Structure

```.
├── Liar.csv  #LIAR Dataset used to train the Classical ML and BERT Models
├── Test_dataset(FINAL).csv #Our new evaluation dataset of curated and synthetic headlines
├── classical_ml_EVAL
│   ├── Classical_ml_EVAL.csv #Predictions from our Classical ML models on the evaluation dataset
│   └── *.ipynb #Notebooks used to train and test out classical baseline models
│ 
├── classical_ml_LIAR
│   └── *.ipynb
│
├── BERT_EVAL
│   ├── BERT_eval.csv #Predictions from our fine-tuned BERT model on the evaluation dataset
│   └── BERT_eval.ipynb #Notebook to compute baseline BERT predictions and results.
│ 
├── Pipeline_Article.ipynb
├── Pipeline_QNA.ipynb
├── Pipeline_SLM(Mistral).ipynb
├── Pipeline_SLM(Phi).ipynb
│ 
├── FactCC_Results #Contains the results for our pipelines utilizing FactCC as the NLI model
│   ├── Pipeline_Article.csv
│   ├── Pipeline_QNA.csv
│   ├── Pipeline_SLM(Mistral).csv
│   └── Pipeline_SLM(Phi).csv
├── Pipeline_Results_FactCC.ipynb #Computation of metrics for the pipelines utilizing FactCC
│ 
├── SummaC_Results  #Contains the results for our pipelines utilizing SummaC (ZS and Conv) as the NLI model
│   ├── Pipeline_Article.csv
│   ├── Pipeline_QNA.csv
│   ├── Pipeline_SLM(Mistral).csv
│   └── Pipeline_SLM(Phi).csv
├── Pipeline_Results_SummaC.ipynb #Computation of SummaC threshold and metrics for the pipelines utilizing SummaC(ZS and Conv)
│ 
├── Efficiency test
│   ├── Efficiency_Test.ipynb #Computes the average execution time for each step of our pipeline
│   └── *.csv #Contain the results for the execution times for each of our different pipelines and their configurations
│
├── unique_decisions.ipynb #Used to generate the venn-diagram plots of unique correct-incorrect decisions
├── scraping_selenium.py #Contains selenium function used for web-scraping in the QNA and LLM pipelines
└── requirements.txt
```

## Illustrating the workflow of our three proposed pipelines with an input headline.

![Veritas_Arch](https://github.com/user-attachments/assets/bc295153-862a-416f-b6e4-fd6cad471c72)

Preprint - https://arxiv.org/abs/2410.09455
