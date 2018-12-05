#!/bin/sh

DOC_REP=bow
MODEL=lr
/media/training-data/hyperpartisam
python train.py -i data/articles-training-bypublisher.xml -o model/ -d $DOC_REP -m lr -l data/ground-truth-training-bypublisher.xml
python predict.py -i data/articles-validation-bypublisher.xml -o predictions -m model/logisticRegression.sav -d $DOC_REP
python evaluator.py -d eval -r predictions -o result 