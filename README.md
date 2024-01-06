# Image classification template 

```
All experiments are tracked with clearml: https://clear.ml/docs/latest/docs/

Environment created with poetry: https://python-poetry.org/docs/
```

1. Setup ClearML: clearml-init

2. Migrate dataset to ClearML: make migrate_dataset

## Dataset information
The dataset was taken from 2018 Data Science Bowl.
The goal is to find the nuclei in divergent images.

https://www.kaggle.com/c/data-science-bowl-2018/overview

## All training and test metrics traced with clearml:
https://app.clear.ml/projects/32a733134a5743739ce9d6719b94c5b3/experiments/2c8c72139fd74d17bbdd035d5a2583f8/output/execution

## Test data segmentation mask example

![alt text](https://github.com/ArtemVerbov/ImageClassification/blob/main/media/confusion_matrix.png?raw=true)
