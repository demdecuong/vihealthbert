# Name Entity Reconition

- We introduce a ViHealthBERT, the first domain-specific pre-trained language model for Vietnamese healthcare
- The performance of our model shows
strong results while outperforming the general domain language models in all health-related datasets. 



Details of our ViHealthBERT model architecture, dataset construction and experimental results can be found in our [following paper]():

    @article{vihealthbert,
        title     = {{ViHealthBERT: Pre-trained Language Models for Vietnamese in Health Text Mining}},
        author    = {Minh Phuc Nguyen, Vu Hoang Tran, Vu Hoang, Ta Duc Huy, Trung H. Bui, Steven Q. H. Truong },
        journal   = {13th Edition of its Language Resources and Evaluation Conference},
        year      = {2022}
        }
**Please CITE** our paper whenever our dataset or model implementation is used to help produce published results or incorporated into other software.



## Model installation, training and evaluation

### Installation
- Python version >= 3.6
- PyTorch version >= 1.4.0

```
    git clone https://github.com/demdecuong/vihealthbert.git
    cd code/Finetune/NER/src
    pip3 install -r requirements.txt
```

### Training and Evaluation
Run the following two bash files to reproduce results presented in our paper:
```
    bash main.sh
```
