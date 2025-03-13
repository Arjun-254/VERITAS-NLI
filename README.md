# VERITAS-NLI : Validation and Extraction of Reliable Information Through Automated Scraping and Natural Language Inference
This repository contains the artifacts for our paper titled `VERITAS-NLI : Validation and Extraction of Reliable Information Through Automated Scraping and Natural Language Inference`, accepted and published in [EAAI - Engineering Applications of Artificial Intelligence](<https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence>).

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
├── SentenceLevelPred.ipynb #Explainability Module
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

## Explainability Module

![Explainability Module Illustration](https://github.com/user-attachments/assets/67144207-c886-42d6-8e32-be9a0641bec4)

## Hyperparameters

### Classical Machine Learning

<table>
<tr>
<td>

| Classifier           | Parameter         | Default Value     | Description                     |
|----------------------|-------------------|-------------------|---------------------------------|
| **Linear SVC**       | penalty           | `'l2'`            | Regularization type             |
|                      | loss              | `'squared_hinge'` | Loss function                   |
|                      | dual              | `'auto'`          | Dual or primal formulation      |
|                      | tol               | `0.0001`          | Tolerance for stopping criteria |
|                      | max_iter          | `1000`            | Maximum iterations              |
| **Logistic Regression** | penalty        | `'l2'`            | Regularization type             |
|                      | dual              | `False`           | Dual or primal formulation      |
|                      | tol               | `0.0001`          | Tolerance for stopping criteria |
|                      | solver            | `'lbfgs'`         | Algorithm for optimization      |
|                      | max_iter          | `1000`            | Maximum iterations              |
|                      | warm_start        | `False`           | Reuse previous solution         |

</td>
<td>

| Classifier           | Parameter         | Default Value     | Description                     |
|----------------------|-------------------|-------------------|---------------------------------|
| **Multinomial NB**   | alpha             | `1.0`             | Additive smoothing parameter    |
|                      | force_alpha       | `True`            | Force alpha parameter for all features |
|                      | fit_prior         | `True`            | Learn class prior probabilities |
|                      | class_prior       | `None`            | Prior probabilities of classes  |
| **Random Forest**    | n_estimators      | `100`             | Number of trees                 |
|                      | criterion         | `'gini'`          | Function to measure split quality |
|                      | min_samples_split | `2`               | Minimum samples to split node   |
|                      | min_samples_leaf  | `1`               | Minimum samples at leaf node    |
|                      | max_features      | `'sqrt'`          | Number of features for best split |
|                      | bootstrap         | `True`            | Bootstrap samples               |

</td>
</tr>
</table>


### Bidirectional Encoder Representations from Transformers (BERT)
- **`num_train_epochs` = 1**: Given the high representational power of BERT and its pre-trained nature, a single epoch is often sufficient to fine-tune the model. Experimental results found that using a greater number of epochs leads to model overfitting.
- **`per_device_train_batch_size` = 16** and **`per_device_eval_batch_size` = 32**: Smaller batch sizes for training help in maintaining a balance between memory usage and effective learning.
- **`warmup_steps` = 100**: Implementing a warmup phase at the beginning of training where learning rates are gradually increased helps in stabilizing the learning process, preventing the model from converging too quickly to a sub-optimal solution.
- **`learning_rate` = 5e-5**: This Learning rate is found to be most suitable for many models for the Adam Optimizer which was used in this particular model training.
- **`weight_decay` = 0.001**: This helps in regularizing the model and preventing over-fitting, which is crucial for a model as large as BERT.
- **`fp16` = True**: This enables the model to train faster and consume less memory while maintaining the training precision, making it feasible to train larger models or use larger batch sizes.

### Natural Language Inference - Summary Consistency (SummaC) 

<table>
<tr>
<td>

| Parameter     | Model ZS                  | Model Conv                                   |
|---------------|---------------------------|---------------------------------------------|
| **Granularity**    | `sentence`           | `sentence`                                  |
| **Model Name**     | `vitc`               | `vitc`                                      |
| **Device**         | `cuda`               | `cuda`                                      |
| **Bins**           | -                    | `percentile`                                |
| **NLI Labels**     | -                    | `e`                                         |
| **Start File**     | -                    | `default`                                   |
| **Aggregation**    | -                    | `mean`                                      |

</td>
<td>

| Pipeline  | Model         | Best Threshold |
|-----------|---------------|-----------------|
| Article   | `model_zs`    | -0.03          |
|           | `model_conv`  | 0.55           |
| QNA       | `model_zs`    | -0.15          |
|           | `model_conv`  | 0.21           |
| Mistral   | `model_zs`    | -0.23          |
|           | `model_conv`  | 0.21           |
| Phi3      | `model_zs`    | -0.13          |
|           | `model_conv`  | 0.21           |

</td>
</tr>
</table>
