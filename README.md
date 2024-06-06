# PND

## Introduction
The **PND** is a multi-corpus dataset to enhance the research of automatic patent novelty detection. 

## Sample Distribution
The `samples` folder contains examples in both Chinese and English. It includes 4,627 positive examples, negative examples, and unrelated examples in Chinese, and 7,488 positive examples, negative examples, and unrelated examples each in English.

## Dataset Description
The `documents.xx.zip` compressed package contains a collection of Chinese and English documents. It comprises 18,597 Chinese documents and 29,863 English documents. Each document sample is formatted as follows:
```python
{
    "pnum": "US3337466A",
    "title": "Effervescent dental cleaner compositions",
    "abstract": "United States Patent Oil-ice 3,337,466 Patented Aug.",
    "description": "United States Patent Oil-ice 3,337,466 Patented Aug. 22, 1967 This invention relates in general to compositions of matter and processes for using the same to produce aerobic conditions"
}
```
## Code Files
**run_tuple.py**: This Python code file is dedicated to binary classification experiments.
**run_triple.py**: This Python code file is dedicated to three classification experiments.
