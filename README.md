#### Table of contents
1. [Introduction](#introduction)
2. [Experimental Results](#result)

# <a name="introduction"></a> ViHealthBERT: Pre-trained Language Models for Vietnamese in Health Text Mining

<img src="asset/overview.png"/>


ViHealthBERT is a strong baseline language models for Vietnamese in Healthcare domain.

 - We empirically investigate our model with different training strategies, achieving state of the art (SOTA) performances on 3 downstream tasks: NER (COVID-19 & ViMQ), Acronym Disambiguation, and Summarization.

 - We introduce two Vietnamese datasets: the acronym dataset (acrDrAid) and the FAQ summarization dataset in the healthcare domain. Our acrDrAid dataset is annotated with 135 sets of keywords.

Our work can be found in this [paper]() - We will update the link whenever the publishing process is completed.


## <a name="result"></a> Experimental Results


| Model         |       Mac-F1*     |   Mic-F1*     |   Mac-F1**    |   Mic-F1**  |
| :---          |       :----:      |    :----:     |    :----:     |   :----:    |  
| PhoBERT-base  |       0.942       |   0.920       |   0.847       |   0.8224    |
| PhoBERT-large |       0.945       |   0.931       |   0.8524      |   0.8257    |
| ViHealthBERT  |       0.9677      |   0.9677      |   0.8601      |   0.8432    |

The overview of experimental results in COVID-19 and ViMQ datasets. * refers to COVID-19 dataset, ** refers to ViMQ dataset.
