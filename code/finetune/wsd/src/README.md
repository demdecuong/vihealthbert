# AcrBERT: Model classification for acronym disambiguation
- We apply model which was proposed by DeepBlue in AAAI-21-SDU-shared-task-2-AD, that is the recent state-of-the-art model.
- We publish supervised acronym disambiguation dataset for medical vietnamese domain. You can download from [here]()

## Dataset
> Once downloaded dataset. You put it data folder followings:

```
   wsd
    ├── src            
    ├── data                    # Dataset files 
    │   ├── train
    │   │   ├── data.json   
    │   ├── dev
    │   │   ├── data.json  
    │   ├── test
    │   │   ├── data.json
    │   ├── dictionary.json     # Dictionary files. A dictionary of the acronyms and their possible meanings.
    │   ├── gold.json           # Gold files. A answer files from data.json
```


## Installation
```
pip install -r requirements.txt
```
## Training and Eval
Run the following bash file to reproduce results

```
!bash main.sh
```
