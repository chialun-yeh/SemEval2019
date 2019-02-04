
This is the repository for the participation of the [SemEval2019 Task4](https://pan.webis.de/semeval19/semeval19-web/).
The main complexity of the task is that the training data are labeled by the publisher. Although more articles that are published by an extremely left-leaning publisher are to be left and extreme, it's hardly the case that all the articles would be. Therefore, it can be seen as a problem of learning from noisy labels. 

The original idea is to filter out cleaner samples and train a model mainly on the training set and test (or maybe finetune a bit) on the test data. We initially trained a baseline model on half of the test set, which achieved 78% accuracy on the other half. We were hoping that using the training set can further improve the accuracy but so far the results haven't been positive. 

Therefore, the submitted system was the baseline model trained only with the test set. It is quite naive as limited by the size of the manually-labeled data. It's a model that uses pre-trained glove vectors as document representation, and SVM as the classifier. Some pre-processing of the text is done. Using TFIDF is likely to achieve comparable performance.

Use svm_baseline.ipyn to reproduce the result. 

We'll keep updating new results and approaches in the future.

This is part of my master thesis in TU Delft, and in cooperation with [De Persgroep Nederland](https://www.persgroep.nl/) .
