# ARN-Labo5-ImageNet


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
1 couche