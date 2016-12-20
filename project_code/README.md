# README

## Bag-of-words models
Use `classify.py`
For example, to get the results for all three baseline classifiers (Naive Bayes, logistic regression, linear SVM) using unigrams only and selecting only the top 5000 features according to a chi-squared test, you'd use:
`python classify.py --all_categories --verbose --lowercase --lemmatize --remove_stop_words --max_n_gram 1 --chi2_select 5000`
To do the same, but this time to learn and use 5000 principal components, you would use:
`python classify.py --all_categories --verbose --lowercase --lemmatize --remove_stop_words --max_n_gram 1 --pca_select 5000`
To get results for WordNet synset features, retaining words that do not have a corresponding synset, use:
`python classify.py --all_categories --verbose --lowercase --lemmatize --remove_stop_words --wn`
To do the same, but this time ignore words that do not have a corresponding synset, use:
`python classify.py --all_categories --verbose --lowercase --lemmatize --remove_stop_words --wn --ignore`

## CNN
Download the glove word embeddings and save the folder `glove.6B` in the same directory as this file.
Then, run `cnn.py` using `python cnn.py`.

## Contact
If you have any questions, feel free to email me at keithstrickling@gmail.com!
