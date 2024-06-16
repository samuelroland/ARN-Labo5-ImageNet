# Labo 5 - ARN
Auteurs: Felix Breval et Samuel Roland

## Introduction
<!-- 1. Introduction: describe the context of your application and its potential uses, brie y describe how you are going to proceed (methodology), that is, what data are you going to collect (your own pictures of.... , maybe extra-pictures collected from the web, etc), what methods are you going to use (e.g., CNNs, transfer learning) . -->
Notre modèle a pour but de classifier la chaîne Youtube à partir d'une vignette d'une de ses vidéos. Pour l'entrainement, nous allons nous baser sur le modèle existant MobileNetV2 et ses poids synaptiques (transfer learning). Les vignettes ne contenant pas fondamentallement de nouvelle caractéristique par rapport aux images de ImageNet (elles contiennent différents objets physiques, visages, personnes, textes, ...), nous estimons qu'il n'y a pas de nouvelle caractéristique à extraire et donc nous n'allons pas rajouter des couches de convolution. Nous allons juste rajouter nos couches de MLP à la fin pour qu'ils appprennent à classifier nos chaînes Youtube. Nous allons entrainer notre modèle grâce aux vignettes directement récupérée sur Youtube, c'est le plus simple, et après l'entrainement nous évaluerons la performance du modèle, ce qui reste confus pour lui et ce qu'il arrive correctement classifier.

## Le problème
<!-- 2. The problem: describe what are the classes you are learning to detect using a CNN, describe the database you collected, show the number of images per class or a histogram of classes (is it a balanced or an unbalanced dataset? Small or big Data ?) and provide some examples showing the intra-class diversity and the apparent di culty for providing a solution (inter-class similarity). -->

Nous avons sélectionné 3 chaînes Youtube liée à l'informatique:
1. **Underscore_**, chaîne de vulgarisation informatique en français, sous forme de talk show toutes les 2 semaines, crée par le Youtubeur Micode
1. **Computerphile**: chaîne anglophone qui parle de beaucoup de sujets geeks ou techniques dans la sécurité ou des curiosités mathématiques derrière des algorithmes
1. **Linus Tech Tips**: chaîne anglophone qui fait des review de matériels qui contient de la technlogie en tout genre

![vignettes](./imgs/vignettes-sample.png)

Comme on peut le voir en partie sur les vignettes au dessus, elles contiennent

1. **Underscore_**: surtout souvent un bout de titre blanc, régulièrement un fond en teinte bleue foncée, régulièrement le visage de Micode (l'auteur) ou celui de 2-3 cohosts.
1. **Computerphile**: toujours un titre en lettre verte dans une police bien particulière, avec souvent un visage mais jamais le même
1. **Linus Tech Tips**: très souvent la tête de Linus avec un visage d'étonnemment la bouche ouverte ou alors tout son corps, souvent accompagné d'un objet

Nous avons imaginé que certaines de ces caractéristiques qui reviennent régulièrement seraient utilisées par le modèle, nous verrons plus tard avec les résultats si c'est le cas ou non. En terme de variété à l'interne d'une même classe, on voit que c'est bien diversifié: par ex. les vignettes d'Underscore_ ont différents formats (cela a évolué au fil du temps), n'ont pas toujours de visage, ont parfois plusieurs visages, ont parfois Micode, parfois d'autres personnes, parfois il y a un fond bleu très clair, d'autres fois c'est plus léger ou inexistant. En terme de difficulté lié au similarité entre les classes, cela nous parait pas trop compliqué à part qu'il y a souvent des visages expressifs (comme les Youtubeurs aiment bien faire), il y a souvent des logos de géants de la tech et des titres parfois dans les mêmes couleurs (rouge, blanc).

Nous avons téléchargé les dernières **200** vignettes de chaque chaîne pour l'entrainement et pris les **30** suivantes comme ensemble de test. Nous avons développé un petit script Python `thumbnail_dl/script.py` permettant de facilement les récupérer, il suffit de le lancer depuis le dossier `thumbnail_dl` pour qu'il télécharge toutes les images nécessaires. Durant le téléchargement, il y a parfois quelques images qui générent des erreurs et ne sont pas téléchargées mais cela ne concerne que 1 image pour Linus Tech Tips et Computerphile, et 4 pour Underscore_. Nous n'avons pas cherché à utiliser plus d'images car la chaîne Underscore_ étant récente, elle n'a pas plus de 230 vidéos publiées.

![sizes](./imgs/dataset-sizes.png)

Notre dataset est donc équilibré, empêchant d'avoir une classe avec moins d'opportunités d'apprentissage que les autres.

## Préparation des données
<!-- 3. Data preparation: describe the pre-processing steps you needed (e.g., resizing, normalization, ltering of collected images, perhaps you discarded some data). Describe the train, validation and test datasets split. 1 -->

Nous n'avons pas eu besoin de labelliser ni trier nos données, car elles sont toutes déjà adaptée. En terme de préparation, nos images sont mises au même format 224x224 que celles d'ImageNet avec `pad_to_aspect_ratio=True` qui permet d'au lieu de recadrer vers le rectangle des vignettes en prenant le carré au milieu, il dezoom dans l'image jusqu'à que la largeur fasse 224pixels. La zone en haut et bas est rempli de noir. Nous avons changé du mode par défaut pour ne pas perdre les bords droites et gauches des images qui contiennent des informations utiles (comme des visages sur les bords qui seraient coupés en deux autrement). Nous appliquons également une normalisation de chaque canal de chaque pixel a des valeurs entre 0 et 1.
```python
# Sizes used in MobileNetV2
IMG_HEIGHT = 224
IMG_WIDTH = 224
image_preprocesses = Sequential([
    Resizing(IMG_HEIGHT, IMG_WIDTH, pad_to_aspect_ratio=True),
    Rescaling(1. / 255) # normalize values between 0 and 1
])
```

Ensuite pour augmenter nos images, nous utilisons 1 couche pour appliquer de la rotation aléatoire de plus ou moins 45 degrés en remplissant les trous par du noir. Même chose pour un zoom aléatoire, on zoome entre 20% vers l'avant et 50% vers l'arrière. Le but de ces transformations est à terme de mieux supporter l'utilisation en condition réelle. Quand on prend une photo avec un téléphone d'une vignette affichée sur un écran devant nous, on n'aura jamais le même angle que la photo originale, et on sera probablement plus loin (comme si on était dézoomé) c'est pour ça que nous appliquons un dezoom plus fort.
```python
image_augmentations = Sequential([
    RandomRotation(factor=1/8, fill_mode='constant'), # A little bit of rotation (45degrees, fill holes with 0)
    RandomZoom(height_factor=(-0.2, 0.5), fill_mode='constant') # A little bit of zoom out (20% in until 50% out)
])
```

## Conception du modèle
<!-- 4.Model creation: describe how did you proceed to come up with a nal model (model selection methodology, hyper-parameter exploration, cross-validation) -->
Notre modèle finale est défini par
- entrainement sur TODO: epochs
- une taille de batch de 32
- optimizer: RMSProp
- loss function: SparseCategoricalCrossentropy

Notre architecture consiste en toutes les couches non denses de ImageNet + les couches suivantes:
- Global average pooling (puis c'est l'entrée du MLP il nous faut revenir en 1 dimension)
- Dropout de 30%
- Couche dense de 100 neurones utilisant la Relu
- Dropout de 30% à nouveau
- Couche dense de sortie de 3 neurones utilisant la softmax

Cette nouvelle partie compte **128,403** paramètres entrainables. Les couches existantes du MobileNetV2 contient **2,257,984** paramètres mais ceux-ci sont gelés donc non entrainables.

Pourquoi le transfer learning ? Cela nous permet de pouvoir entrainer un modèle avec moins d'images comme on bénéficie déjà de "l'intelligence" du modèle existant à déjà reconnaitre une grande quantité d'objets, les couches de convolution fonctionne déjà bien pour extraire toutes les caractéristiques nécessaires à notre problème. En plus de reprendre l'architecture, en reprenant les poids synaptiques, cela nous évite de devoir réentrainer ces 2 millions de paramètres, ils sont déjà à "valeurs utiles" pour notre problème. On prend donc l'architecture + les poids et on retire les couches de fin (le MLP de MobileNetV2) pour pouvoir ajouter notre propre MLP spécifiques à notre problème (qui a 3 classes et non 1000).

a.b.c.What hyperparameters did you choose (nb epochs, optimizer, learning rate, ...) ?
What is the architecture of yourdoes it have?
nal model ? How many trainable parameters
How did you perform the transfer learning ? Explain why did you use transfer
learning, e.g., what is the advantage of using transfer learning for this problem
and why it might help ?

<!-- 5.6.a.b.c.What hyperparameters did you choose (nb epochs, optimizer, learning rate, ...) ? -->
<!-- What is the architecture of yourdoes it have? -->
<!-- nal model ? How many trainable parameters -->
<!-- How did you perform the transfer learning ? Explain why did you use transfer -->
<!-- learning, e.g., what is the advantage of using transfer learning for this problem -->
<!-- and why it might help ? -->
<!-- Results: describe the experiments you performed with the model both o -line (running -->
<!-- in your notebooks with your own-collected images) and on-line (running in the -->
<!-- smartphone and processing images captured in the “wild”). -->
<!-- a. Provide your plots and confusion matrices -->
<!-- b.c.d.e.f. -->
<!-- Provide the f-score you obtain for each of your classes. -->
<!-- Provide the results you have after evaluating your model on the test set. -->
<!-- Comment if the performance on the test set is close to the validation -->
<!-- performance. What about the performance of the system in the real world ? -->
<!-- Present an analysis of the relevance of the trained system using the Class -->
<!-- Activation Map methods (grad-cam) -->
<!-- Provide some of your misclassi ed images (test set and real-world tests) and -->
<!-- comment those errors. -->
<!-- Based on your results how could you improve your dataset ? -->
<!-- g.Observe which classes are confused. Does it surprise you? In your opinion, what can cause those confusions ? What happens when you use your embedded system to recognize objects that don’t belong to any classes in your dataset ? How does your system work if your object is placed in a di erent background ? -->
<!-- Conclusions: nalize your report with some conclusions, summarize your results, mention the limits of your system and potential future work. -->
<!-- 

notes random

basic


layers = Dense(128, activation='relu')(layers)
layers = Dropout(0.5)(layers)
3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 727ms/step - acc: 0.9115 - loss: 0.6417 - val_acc: 0.8750 - val_loss: 0.8740


# adding some dense layers here
layers = Dense(128, activation='relu')(layers)
layers = Dense(128, activation='relu')(layers)
layers = Dropout(0.5)(layers)

testacc
4/4 ━━━━━━━━━━━━━━━━━━━━ 2s 521ms/step - acc: 0.9010 - loss: 1.0554
normacc
3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 811ms/step - acc: 0.9740 - loss: 0.2183 - val_acc: 0.7500 - val_loss: 1.8775

layers = Dense(128, activation='relu')(layers)
layers = Dense(128, activation='relu')(layers)
3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 746ms/step - acc: 0.9831 - loss: 0.0377 - val_acc: 0.8750 - val_loss: 0.7675
testacc 4/4 ━━━━━━━━━━━━━━━━━━━━ 2s 520ms/step - acc: 0.9946 - loss: 0.0064  


second run - le best - commité
3/3 ━━━━━━━━━━━━━━━━━━━━ 2s 847ms/step - acc: 1.0000 - loss: 9.7941e-06 - val_acc: 0.8750 - val_loss: 1.2603
4/4 ━━━━━━━━━━━━━━━━━━━━ 2s 544ms/step - acc: 1.0000 - loss: 0.0049


Felix, [17/05/2024 16:35]
3/3 ━━━━━━━━━━━━━━━━━━━━ 3s 958ms/step - acc: 0.9831 - loss: 0.0718 - val_acc: 0.9167 - val_loss: 0.4893

Felix, [17/05/2024 16:35]
juste 128, pas de dropout

Felix, [17/05/2024 16:38]
1 couche -->

