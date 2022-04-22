# Name Entity Reconition

### Installation
- Python version >= 3.6
- PyTorch version >= 1.4.0

```
    git clone https://github.com/demdecuong/vihealthbert.git
    cd code/finetune/ner/src
    pip3 install -r requirements.txt
```
### Data
- Dowload COVID-19] (word-version) and ViMQ,
- Copy train/dev/test data into `data/vinai_covid_word` and `data/vimq_word` respectively
- Preprocess data into IOB format:
```
python preprocess.py
```
### Training and Evaluation
Run the following two bash files to reproduce results presented in our paper:
```
    bash main.sh
```
