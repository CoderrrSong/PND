# PND

## Introduction
The **PND** is a multi-corpus dataset to enhance the research of automatic patent novelty detection. 

## Sample Distribution
The `samples` folder contains examples in both Chinese and English. It includes 4,627 positive examples, negative examples, and unrelated examples in Chinese, and 7,488 positive examples, negative examples, and unrelated examples each in English.

## Dataset Description
The `documents.xx.zip` compressed package contains a collection of Chinese and English documents. These documents contain field information for all patents related to the data in the samples folder. It comprises 18,597 Chinese documents and 29,863 English documents. Each document sample is formatted as follows:
```python
{
"pnum": "US3337466A",
"title": "Effervescent dental cleaner compositions",
"abstract": "United States Patent Oil-ice 3,337,466 Patented Aug. 22, 1967 This invention relates in general to compositions of matter and processes for using the same to produce aerobic conditions, antiseptic activity, bleaching effects and detergent action and combinations Olf these activities.",
"description": "United States Patent Oil-ice 3,337,466 Patented Aug. 22, 1967 This invention relates in general to compositions of matter and processes for using the same to produce aerobic conditions, antiseptic activity, bleaching effects and detergent action and combinations Olf these activities. The invention relates in particular to compositions and processes for the cleansing, sterilizing and bleaching of dentures. This application is a continuation-in-part of our co-pending United States patent application, Ser. No. 185,246, filed April 5, 1962 now abandoned.Therefore, it is a general object of the present invention to provide a composition and process for cleansing, sterilizing and bleaching in a simple and efiicacious manner.",
"claims": "1. A DENTURE CLEANER COMPOSITION CONSISTING OF BY WEIGHT FROM ABOUT 5 TO 40 PARTS OF A MIXTURE CONSISTING OF ABOUT 50 MOLE PERCENT POTASSIUM MONOPERSULFATE, ABOUT 25 MOLE PERCENT POTASSIUM SULFATE, AND ABOUT 25 MOLE PERCENT POTASSIUM HYDROGEN SULFATE, FROM ABOUT 40 TO 5 PARTS OF AN INORGANIC WTER SOLUBLE PEROXIDE OF METAL SELECTED FROM THE GROUP CONSISTING OF GROUPS I AND II OF THE PERIODIC TABLE, UP TO ABOUT 5 PARTS OF A WATER SOLUBLE HALIDE SELECTED FROM THE GROUP CONSISTING OF THE CHLORIDE, BROMIDE AND IODIDE OF THE ALKALI METALS AND ALKALINE EARTH METALS AND OF AMMONIUM; "
}
```
## Code Files
**run_tuple.py**: This Python code file is dedicated to binary classification experiments.
**run_triple.py**: This Python code file is dedicated to three classification experiments.
