# Reconnaissance de cellules infectées par la Malaria

Une adolescente a créé un réseau de neurones capable de reconnaître des cellules infectées par la Malaria ou non.
Elle a utilisé les modules Tensorflow et Keras en python pour créer son réseau de neurones.

## Stage IA à Magic Makers

[Magic Makers](https://www.magicmakers.fr/) propose des ateliers de programmation créative pour des jeunes de 7 à 15 ans. Depuis 2018, des ateliers pour adolescents autour de l'intelligence artifielle sont donnés durant les vacances. Lors du stage, les makers découvrent ce qu'est un réseaux de neurones et les notions s'y attachant (perceptron multi-couches, convolutions, overfit, etc) en créant des projets comme celui-ci !

## Auteur du projet

Ce projet a été réalisé par **Lisa** (15 ans) lors du stage de Février dans le centre de Magic Makers Paris 9e, animé par **Jade, Emilie et Antoine**.


### Dataset

* [Dataset de cellules](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria) - Images de cellules infectées et non infectées par la Malaria


### Entraînement

Lisa a utilisé un réseau de neurones par convolution pour son projet.

```
python3 Lisa-cell-train.py
```
## Modules

* [Keras](https://keras.io/) - pour créer le modèle (avec TensorFlow)
* [Flask](http://flask.pocoo.org/) - pour créer une webapp
* [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html) - pour manipuler des images
* [Numpy](https://www.numpy.org/) - pour manipuler des tableaux
* [H5py](https://www.h5py.org/) - pour sauvegarder le modèle
* [Sklearn](https://scikit-learn.org/stable/) - pour mélanger et séparer les données


## Résultats

< à venir >

### Application

Une fois son modèle entraîné, Lisa a poursuivi son projet en en faisant une webapp qui permet d'uploader une image et d'obtenir la prédiction de son modèle sur cette image !

Sous Mac et Linux :
```
export FLASK_ENV = development
export FLASK_APP = Lisa-cell-webapp.py

flask run
```

Sous Windows :
```
set FLASK_ENV = development
set FLASK_APP = Lisa-cell-webapp.py

flask run
```

### Remerciement

* * Merci à [Kaggle](https://www.kaggle.com/) pour le dataset
* Merci à [Magic Makers](https://www.magicmakers.fr/)
* Merci à [Keras](https://keras.io/) pour faciliter la création de réseaux de neurones !
