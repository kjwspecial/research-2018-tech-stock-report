# research-2019-Stock-Price_Technical_Analysis_Report

Stock-to-text System

# Requirement
* Python
* numpy
* pandas
* tensorflow '1.12.0'
* sklearn
* nltk
* gensim

# Word Embedding
```bash
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ mkdir build && cd build && cmake ..
$ make && make install
$ ./fasttext skipgram -input ../../txt_data/analysis_data.txt -output ../../fasttext_vector -dim 256 -ws 3 -epoch 50 -minCount 20
```
