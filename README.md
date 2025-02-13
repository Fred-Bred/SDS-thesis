# The Language of Attachment: Modelling Attachmetn Dynamics in Psychotherapy
This repository contains LaTeX files and code for the implementation of language models, primarily RoBERTa variants, trained to classify psychotherapy transcripts according to attachment styles. The goal is to provide an automated tool for analysing therapeutic interactions and understanding client attachment patterns. All work was conducted as part of my master's thesis in social data science.

## Implementation details
For more details on experiments, results, and implications, see the thesis ('TeX/main.pdf'). There, you will also find an introduction to attachment theory and a review of the literature underscoring its relevance to psychotherapy.

This project tested the feasibility of automatically classifying psychotherapy patients into one of three organised attachment patterns based on their in-session utterances. In practice, this was done by training various models, mainly RoBERTa variants, to classify sections of client-only speech.

Data and labels were obtained from previous research conducted by Talia et. al (2017) in their validation study of the Patient Attachment Coding System (PACS).

Most models were implemented using [MaChAmp](https://github.com/machamp-nlp/machamp). The majority of files for this implementation were hosted only on the server used for secure data storage and computing and are not included in this repository.

Experiments primarily centered around
- speech turn length, i.e. the amount of context afforded for each classification,
- model version and capacity
- pre-training

Due to its sensitive nature, data cannot be made available.

## Thesis Abstract
Early childhood attachments lay the foundation for future psychological wellbeing
by forming mental models of one-self and others. Attachment styles
predictably affect individuals’ perceptions and interactions with their environment.
This, in turn, influences a breadth of life outcomes as well as the
development and treatment of psychopathologies. In therapy, insight into a
patient’s attachment characteristics can inform the organisation of treatment
or represent the primary objective. However, the assessment of attachment
style is resource intensive and the best tools do not lend themselves well
to repeated measurement. This thesis aims to investigate the feasibility of
new, automated methods driven by machine learning in assessing patient
attachment characteristics from psychotherapy transcripts. It builds on assumptions
and data from the Patient Attachment Coding System (Talia,
Miller-Bottome, & Daniel, 2017) to develop an automated pipeline robust to
therapist characteristics and therapeutic modality. Fine-tuned RoBERTalarge
encoders for classification reach a mean test set accuracy of 59.55 %
(SD = 5.82) in distinguishing between the three classes of anxious, secure,
and avoidant attachment styles. Results highlight the complexity of the
phenomenon of attachment and direct the focus of future research towards
including more context, more capable base models, and larger training data
sets. Implications for research and practice are discussed and ethical guidelines
for the first steps in deployment are offered.