# README

## Contribution

Online resources: 
- TabNet - Optuna for fine-tuning (from [optuna-examples](https://github.com/optuna/optuna-examples/blob/main/tensorflow/tensorflow_estimator_simple.py))
- TabNet - FeatureBlock (from [Ghost Batch Normalisation](https://github.com/ostamand/tensorflow-tabnet/blob/master/tabnet/models/gbn.py))

- TabNet - Architecture (from [Codebase for "TabNet: Attentive Interpretable Tabular Learning"](https://github.com/google-research/google-research/tree/master/tabnet#codebase-for-tabnet-attentive-interpretable-tabular-learning))

Team members:

- LDA:
- XGBoost:
- TabNet:

## Code documentation

- `Baseline.ipynb`: code for presentation 2
- `Model.ipynb`: code for ML part of presentation 3
- 


## Overview

The purpose of the code is to digest large set of financial statements in pdf format and try to exrtact the most relevant part of the statement and then pass the result into ChatGPT to work out the 18 features which can be used by the machine learning models TabNet and XGBoost

The key elements in the package for NLP part consist of:

#1. LDA model "COMPSCI760_pretrained_model" jupyter source file

#2. LDA visualisation tool"COMPSCI760_Visualisation" jupyter source file

#3. AccessAPI Python file "COMPSCI760_accessapi" to demonstrate the ChatGPT prompt and output

#4. Data folder used by both the LDA part and ChatGPT part.

 (from [Codebase for "Machine Learning Samples"](https://github.com/Azure-Samples/MachineLearningSamples-DocumentCollectionAnalysis/tree/master/Code))

The methods presented in this project allowed us to witness the potential that can be and should further developed given more time and resource in the future. I can envisage the topic modelling can be finetuned with more domain knowledge built in to attain better results in terms of accuracy and efficiency. 

The machine learning techniques/algorithms used in this project include:

1. Text processing and cleaning

2. Topic modeling

3. Result summmarisation

4. LDA visualisation 


## Prerequisites

The prerequisites to run this example are as follows:

* Make sure that you your own ChatGPT API key and at the moment I used my own private ChatGPT API key.

* Make sure you run the code on Visual Basic Studio Python 3.11.6 64-bit.Windows
