# Deploy a ML model with fastAPI and Docker

This is a simple example to deomnstarte how to deploy a webserver that hosts a predictive model trained on the [wine dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html#sklearn.datasets.load_wine) using [FastApi](https://fastapi.tiangolo.com/) and [Docker](https://www.docker.com/).

It leverages FastAPI's webserver functionalities to deploy a Deep Learning model and integrate the code with Docker so it is portable and can be deployed with more ease.

The model is a simpler classifier that consists of a [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and a [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) with a forest of 10 trees.

Within the documentation, snippets of the files will be displayed with a description of what is going on.
