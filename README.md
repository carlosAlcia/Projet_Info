# Projet Option Info : Learning an object-centric world-model with a low-cost robotic arm

Ce repo contient le code modifié de Dino-WM utilisé pour les expériences réalisées dans le cadre de notre projet d'option : Learning an object-centric world-model with a low-cost robotic arm. 

Le code original (et nos modifications), sont tous 2 très "sales", mais le but de ce README est d'y voir un peu plus clair.

Si vous êtes sur Windows, il faut utiliser Anaconda dans WSL. Pour créer et activer l'environnement : 
```
conda env create -f dino_wm/environment.yaml
conda activate dino_wm
```

Il y a 3 fichiers à la racine de ce dépôt qui sont les "exécutables finaux" et serviront de points d'entrées aux explications : `read_ds.py`, `train.sh`et `plan.sh`. Si vous êtes sur windows, `read_ds.py` est à lancer sur windows (explication plus loin) et `train.sh`et `plan.sh` sur WSL. Pour rappel, pour exécuter des fichiers bash sous Linux:
```
chmod +x file.sh # donne la permission d'exécution, à faire une fois
./file.sh
```

## `read_ds.py`

Ce script Python permet de convertir une base de donnée (stockée sous forme de fichier) au format LeRobot en base de donnée utilisable par Dino-WM. Le script suppose que la base de donnée LeRobot est stockée à l'endroit par défaut après avoir été fetch en utilisant LeRobotDataset(username/idDataset) dans un autre script, c'est à dire `User/.cache/huggingface/LeRobot/`. Le script localise bien ce dossier sur Windows et sur Linux, mais sur WSL, le dossier "home" de l'utilisateur n'est pas le même que dans Windows. Il faut donc soit installer LeRobot dans l'environnement conda dino-wm sur WSL, fetch le dataset puis utiliser le script (le tout dans WSL), soit utiliser le script dans Windows (si vous travailler avec LeRobot sur Windows) puis poursuivre dans WSL.

La variable `dataset_name` permet de choisir le dataset à convertir. Sa valeur est du type "huggingface_username/dataset_name". Le dataset au format dino-wm sera stocké dans le répertoire `dataset_dino/custom`. Ces 2 termes sont importants et ne peuvent pas être renommés à la volée : il faudrait modifer "dataset_dino" dans `train.sh` et "custom" dans `dino_wm/conf/env/custom_env.yaml`. En effet, pour utilisation d'un dataset de données réelles, il nous a fallu créer notre propre environnement "custom" (décrit par `custom_env.yaml`) qui fait appel à la classe `CustomDataset` dans `dino_wm/datasets/custom_dset.py` qui lira les fichiers stockés pour charger les données en mémoire, appliquera les pré-traitement, et implémente les fonctions permettant d'accéder aux données, de la même manière que pour les autres types de dataset.

## `train.sh`

Ce script bash va exporter 3 variables d'environnement : `DATASET_DIR`, `WANDB_MODE` et `HYDRA_FULL_ERROR`, et lancer le script python `dino_wm/train.py`. `DATASET_DIR` donne le chemin jusqu'au jeu de donnée (produit par conversion avec `read_ds.py`). `WANDB_MODE` est à "online" par défaut, mais si Weights and Biases pose problème, on peut remplacer "online" par "disabled" pour le désactiver. `HYDRA_FULL_ERROR=1` permet d'avoir des messages d'erreur plus complets.

Enfin, le script `dino_wm/train.py` réalise l'entrainement lui-même. Les résultats de l'entrainement sont mis dans le dossier `outputs/VWM`. L'entrainement est paramétré par le fichier `dino_wm/conf/train.yaml`. On remarquera la ligne qui choisi notre (faux) environnement `env: custom_env` (et donc le custom dataset, ce qui nous permet de charger les données)


## `plan.sh`


## Expériences

Les résultats d'entraînement sur la base de donnée : https://huggingface.co/datasets/Ityl/so100_recording2 sont disponibles sur https://huggingface.co/Ityl/Dino-WM.