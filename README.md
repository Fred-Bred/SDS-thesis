# The Language of Attachment: Modelling Attachmetn Dynamics in Psychotherapy
This repository contains LaTeX files and code for the implementation of language models, primarily RoBERTa variants, trained to classify psychotherapy transcripts according to attachment styles. The goal is to provide an automated tool for analysing therapeutic interactions and understanding client attachment patterns. All work was conducted as part of my master's thesis in social data science.

## Implementation details
For more details on experiments, results, and implications, see the thesis ('TeX/main.pdf'). There, you will also find an introduction to attachment theory and a review of the literature underscoring its relevance to psychotherapy.

This project tested the feasibility of automatically classifying psychotherapy patients into one of three organised attachment patterns based on their in-session utterances. In practice, this was done by training various models, mainly RoBERTa variants, to classify sections of client-only speech.

Data and labels were obtained from previous research conducted by Talia et. al in their validation study of the Patient Attachment Coding System (PACS).

Most models were implemented using [MaChAmp](https://github.com/machamp-nlp/machamp). The majority of files for this implementation were hosted only on the server used for secure data storage and computing and are not included in this repository.

Experiments primarily centered around
- speech turn length, i.e. the amount of context afforded for each classification,
- model version and capacity
- pre-training