# WSA-BERT
Word-wise Sparse Attention for Multimodal Sentiment Analysis
## First check that the requirements are satisfied:
* Python 3.6.5
* torch 1.4.0
* scikit-learn 0.20.3
* tqdm 4.62.2
* numpy 1.19.4
* **transformers** 3.0.2
## The next step is to clone the repository:
```
$ git clone https://github.com/qianfan1996/WSA-BERT.git
```
## Then create data/, saved_models/ directory
```
$ mkdir data, saved_models
```
## Put CMU-MOSI and CMU-MOSEI datasets into data/ directory
## You can run the code with:
```
$ CUDA_VISIBLE_DEVICES=0 python main.py --model WA-BERT --dataset mosi
```
in the command line. In addition, you can also change command line arguments to train different models on different datasets

run the code with:
```
$ CUDA_VISIBLE_DEVICES=0 python bert_classifier.py --dataset mosi
```
This will only use text to classify the sentiment.
# Acknowledge
We acknowledge [this great repository](https://github.com/WasifurRahman/BERT_multimodal_transformer) for reference.
