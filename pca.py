'''Principal component analysis'''


# basically:

pca = PCA(n_components=best_num_components)

# fit the PCA to X_train so that we get the same transformation for X_test later on
pca.fit(X_train)

# update X_train
X_train = pca.transform(X_train)

# Then use the updated X_train to train the model
clf.fit(X_train, y_train)


# Then later on,
X_test = pca.transform(X_test)
clf.predict(X_test)
