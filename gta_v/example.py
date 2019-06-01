from pathlib import Path

import time
import cv2

import server


def main():
    env = server.GTAVEnvironment(blocking=False)
    env.init()

    while True:
        o = env.step()
        t = o['timestamp']

        cv2.imshow('final', o['final'])
        cv2.waitKey(1)

        time.sleep(1)


if __name__ == '__main__':
    main()
