import os

import numpy as np
import vizdoom as vzd

import cv2


DEBUG = 0
DEBUG_COLORS = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
        ]


WALL_ID = 0
FLOOR_CEILING_ID = 1

ITEM = 0
ENEMY = 1
OTHER = 2
WALL = 3
FLOOR = 4
CEILING = 5

N_SEMANTICS = 6

NAME_TO_LABEL = dict()
NAME_TO_LABEL['DoomPlayer'] = OTHER
NAME_TO_LABEL['ClipBox'] = ITEM
NAME_TO_LABEL['RocketBox'] = ITEM
NAME_TO_LABEL['CellPack'] = ITEM
NAME_TO_LABEL['RocketLauncher'] = ITEM
NAME_TO_LABEL['Stimpack'] = ITEM
NAME_TO_LABEL['Medikit'] = ITEM
NAME_TO_LABEL['HealthBonus'] = ITEM
NAME_TO_LABEL['ArmorBonus'] = ITEM
NAME_TO_LABEL['GreenArmor'] = ITEM
NAME_TO_LABEL['BlueArmor'] = ITEM
NAME_TO_LABEL['Chainsaw'] = ITEM
NAME_TO_LABEL['PlasmaRifle'] = ITEM
NAME_TO_LABEL['Chaingun'] = ITEM
NAME_TO_LABEL['ShellBox'] = ITEM
NAME_TO_LABEL['SuperShotgun'] = ITEM
NAME_TO_LABEL['TeleportFog'] = OTHER
NAME_TO_LABEL['Zombieman'] = ENEMY
NAME_TO_LABEL['ShotgunGuy'] = ENEMY
NAME_TO_LABEL['HellKnight'] = ENEMY
NAME_TO_LABEL['MarineChainsawVzd'] = ENEMY
NAME_TO_LABEL['BaronBall'] = ENEMY
NAME_TO_LABEL['Demon'] = ENEMY
NAME_TO_LABEL['ChaingunGuy'] = ENEMY
NAME_TO_LABEL['Blood'] = OTHER
NAME_TO_LABEL['Clip'] = ITEM
NAME_TO_LABEL['Shotgun'] = ITEM

NAME_TO_LABEL['CustomMedikit'] = ITEM
NAME_TO_LABEL['DoomImp'] = ENEMY
NAME_TO_LABEL['DoomImpBall'] = ENEMY
NAME_TO_LABEL['BulletPuff'] = OTHER
NAME_TO_LABEL['Poison'] = ENEMY

NAME_TO_LABEL['BurningBarrel'] = OTHER
NAME_TO_LABEL['ExplosiveBarrel'] = OTHER
NAME_TO_LABEL['DeadExplosiveBarrel'] = OTHER
NAME_TO_LABEL['Column'] = OTHER
NAME_TO_LABEL['ShortGreenTorch'] = OTHER


def transform_labels(labels, labels_buffer):
    n = labels_buffer.shape[0] // 2
    semantics = np.zeros(labels_buffer.shape + (N_SEMANTICS,), dtype=np.uint8)
    semantics[labels_buffer==WALL_ID,WALL] = 1
    semantics[labels_buffer==FLOOR_CEILING_ID,FLOOR] = 1
    semantics[:n,:,FLOOR] *= 0
    semantics[labels_buffer==FLOOR_CEILING_ID,CEILING] = 1
    semantics[n:,:,CEILING] *= 0

    for label in labels:
        axis = NAME_TO_LABEL.get(label.object_name, OTHER)
        semantics[labels_buffer==label.value,axis] = 1

        if label.object_name not in NAME_TO_LABEL:
            print(label.object_name)

    if DEBUG:
        print(semantics.sum(-1).max())
        # print(list(sorted(set(labels_buffer.reshape(-1)))))

        tmp = np.zeros(labels_buffer.shape + (3,), dtype=np.uint8)

        for i in range(N_SEMANTICS):
            tmp[semantics[:,:,i] == 1] = DEBUG_COLORS[i]

        cv2.imshow('debug_labels', cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    return semantics
