# Fostering_Cyber_Threat_Detection_Through_FL
The source code for the paper *Fostering Cyber Threat Detection Through Federated Learning*.


## Overview

Despite achieving good performance and wide adoption, machine learning based security detection models (e.g., malware classifiers) are subject to concept drift and evasive evolution of attackers, which renders up-to-date threat data as a necessity.  However, due to enforcement of various privacy protection regulations (e.g., GDPR),   it is becoming increasingly challenging or even prohibitive for security vendors to collect individual-relevant and privacy-sensitive threat datasets, e.g., SMS spam/non-spam messages from mobile devices. To address such obstacles, this study systematically profiles the (in)feasibility of federated learning for privacy-preserving cyber threat detection in terms of effectiveness, byzantine resilience, and efficiency. This is made possible by the build-up of multiple threat datasets and threat detection models, and more importantly, the design of realistic and security-specific experiments. 

We evaluate FL on two representative threat detection tasks, namely SMS spam detection and Android malware detection. It shows that FL-trained detection models can achieve a performance that is comparable to centrally trained counterparts. Also, most non-IID data distributions  have either minor or negligible impact on the model performance, while a label-based non-IID distribution of a high extent can incur non-negligible fluctuation and delay in FL training. Then, under a realistic threat model, FL turns out to be adversary-resistant to attacks of both data poisoning and model poisoning. Particularly, the attacking impact of a practical data poisoning attack is no more than 0.14\% loss in model accuracy. Regarding FL efficiency,  a bootstrapping strategy turns out to be  effective to mitigate the training delay as observed in label-based non-IID scenarios.

## Datasets

The datasets in our work are all open source as described in our [Project site](https://chasesecurity.github.io/Fostering_Cyber_Threat_Detection_Through_FL/).

## Code

Our work evaluates FL on two threat detection tasks: SMS spam detection and Android malware detection. And the experiment code can be easily extended to other binary classification tasks.

To try this code, you need to **1. get your dataset ready 2. preprocess your data 3. try the central or federated learning.** More READMEs can be found in subfolders.

Now choose a threat detection task [SMS spam detection](SMSSpam) or [Android malware detection](AndroidMalware) to get started.

### Directory structure

```
.
├── AndroidMalware
│   ├── central_learning
│   ├── data
│   ├── data_preprocessing
│   ├── federated_learning
│   └── requirements.txt
├── SMSSpam
│   ├── central_learning
│   ├── data
│   ├── data_preprocessing
│   ├── federated_learning
│   └── requirments.txt
└── strategy
```
