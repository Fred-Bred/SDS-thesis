# The Language of Attachment: Modelling Attachment Dynamics in Psychotherapy
This repository contains LaTeX files and code for the implementation of language models, primarily RoBERTa variants, trained to classify psychotherapy transcripts according to attachment styles. The goal is to provide an automated tool for analysing therapeutic interactions and understanding client attachment patterns. All work was conducted as part of my master's thesis in social data science.

## Files of interest
### Config and bash files
As the bulk of experiments for this project were conducted using [MaChAmp](https://github.com/machamp-nlp/machamp), this reposity does not contain code for the vast majority of training runs.
However, MaChAmp config files for the main experiments are available in the **src/configs/** directory.
Likewise, bash scripts used to call MaChAmp scripts and implement five-fold cross-validation across five base encoders and five input lengths can be found in **src/bash_scripts/**.
Here, the **run.sh** script should provide an overview of relevant files to look into to understand the methodology.

### Early experiments
For my initial experiments, before implementing the main MaChAmp-powered solution, I ran initial experiments with various input hyper-parameter settings to narrow down relevant ranges and parameters for the main experiments.
These were primarily implemented through variations of the **src/utils/trainer.py** and **src/train.py**.


## Implementation details
For details on experiments, results, and implications, see the thesis ('TeX/main.pdf'). There, you will also find an introduction to attachment theory and a review of the literature underscoring its relevance to psychotherapy.

This project tested the feasibility of automatically classifying psychotherapy patients into one of three organised attachment patterns based on their in-session utterances. In practice, this was done by training various models, mainly RoBERTa variants, to classify sections of client-only speech.

Data and labels were obtained from previous research conducted by Talia et. al (2017) in their validation study of the Patient Attachment Coding System (PACS).

Most models were implemented using [MaChAmp](https://github.com/machamp-nlp/machamp). See the MaChAmp repo for code and further details on this.

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