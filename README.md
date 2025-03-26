# Projet Option Info : Learning an object-centric world-model with a low-cost robotic arm

Ce repo contient le code modifié de Dino-WM utilisé pour les expériences réalisées dans le cadre de notre projet d'option : Learning an object-centric world-model with a low-cost robotic arm. 

Le code original et nos modifications, sont tous 2 très sales, mais le but de ce README est d'y voir un peu plus clair.

Si vous êtes sur Windows, il faut utiliser Anaconda dans WSL. Pour créer et activer l'environnement : 
```
conda env create -f environment.yaml
conda activate dino_wm
```

Il y a 3 fichiers à la racine de ce dépôt qui sont les "exécutables finaux" et serviront de points d'entrées aux explications : `read_ds.py`, `train.sh`et `plan.sh`. Si vous êtes sur windows, `read_ds.py` est à lancer sur windows (explication plus loin) et `train.sh`et `plan.sh` sur WSL. Pour rappel, pour exécuter des fichiers bash sous Linux:
```
chmod +x file.sh # donne la permission d'exécution, à faire une fois
./file.sh
```

## `read_ds.py`
## `train.sh`
## `plan.sh`
