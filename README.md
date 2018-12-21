# Automatic Juridical Decision-making: a Neural Network approach applied to the European Court of Human Rights

## About

In recent years, the potential to speed up legal processes via Machine Learning techniques has increased. Consequently, this study serves to investigate different NLP and ML methods. It does this by building a model that predicts whether or not an article of ECtHR has been violated. The most effective model , which consists of a Neural Network (for embedding the corpus) and a SVM (as classifier), reaches a total mean accuracy of 72%. 

## References

All the details about this project are well explained in this [document](/thesis.pdf). 

## Quick start

You can find all the models in folder "models" , as you can see from the documentation we have 5 different models (you can find a detailed description of them in document mentioned above).
The folder "results" contained the final results of the last model (model 5) and there is a 10x10 matrix for each article, while the folder "medvedeva" contains the results obtained by applying an approach as it is described in this [paper](http://martijnwieling.nl/files/Medvedeva-submitted.pdf). The other notebook [Correletd t-test](https://github.com/daminienrico/automatic_juridical_decision_making/blob/master/Correlated%20t-test%20for%20comparing%20classifiers%20performance%20on%20the%20same%20dataset.ipynb) and [Hierarchical test.ipynb](https://github.com/daminienrico/automatic_juridical_decision_making/blob/master/Hierarchical%20test.ipynb) shows the Bayesian comparison between the two models. 
 
## Requirements 

* numpy
* [BLAS and LAPACK (Reccomanted)](http://www.netlib.org/blas/)
* pandas
* [nltk](https://www.nltk.org/)
* [Gensim](https://radimrehurek.com/gensim/models/doc2vec.html)
* [sklearn](http://scikit-learn.org/stable/)
* [keras](https://keras.io/)
* [Python 3](https://www.python.org/download/releases/3.0/)
* [Pystan](https://pystan.readthedocs.io/en/latest/)
* [Jupyter Notebook (Reccomanted with Anaconda)](http://jupyter.org/)
