
This is the repository for the participation of the [SemEval2019 Task4](https://pan.webis.de/semeval19/semeval19-web/).
The dataset provided for the task consists of the following two parts:

Part 1: by-publisher
- training_1: 600K articles that are labeled by the bias of the source of the article.
- validation_1: 150K articles that are labeled by the bias of the source of the article. The publishers are different from training_1.

Part 2: by-article
- training_2: 645 articles that are labeled to be either hyperpartisan or non-hyperpartisan by Turkers.

In correlation_publisher_bias.ipynb, we show that the publisher bias has certain correlation with article bias. However, a biased publisher can publish both bias and unbiased articels. We mainly experimented with the following features:

1. TFIDF: a tfidf vectorizer trained with the training_1 and validation_1 and a SVM/LR trained with the training_2. You need to have all data and install [scikit-learn](https://scikit-learn.org/stable/)
2. Doc2Vec: a doc2vec model trained with the training_1 and validation_1 and a SVM/LR trained with the training_2. You need to have all data and install [gensim](https://radimrehurek.com/gensim/)
3. Pretrained GloVe: we extract the word vectors of training_2 and average all the words within an document. Then a SVM/LR is trained. You need to have the small dataset and [pretrained GloVe vectors](https://nlp.stanford.edu/projects/glove/)
4. Features that are partly inspired by http://nelatoolkit.science/. Some codes and lexicons are also taken from the toolkit.

We'll keep updating new results and approaches in the future.
This is part of my master thesis in TU Delft, and in cooperation with [De Persgroep](https://www.persgroep.nl/) .
