import pandas as pd
import cv2
import numpy as np

image_path = 'crop_part1\\'
landmark_file = 'landmark_list_part1.txt'

lmks = pd.read_csv(landmark_file, delim_whitespace=True, header=None,
                  names=['name'] + [col+str(i) for i in range(68) for col in ['x', 'y']])


def draw_on_image(image, text, lmk):
    cv2.putText(image, text, (0, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
    for l in lmk:
        cv2.circle(image, (l[0], l[1]), 1, (255, 0, 0), 1)


A20 = []
A85 = []
for i in range(lmks.shape[0]):
    image_file = image_path + lmks.loc[i, 'name'] + '.chip.jpg'
    age = int(lmks.loc[i, 'name'].split('_')[0])
    if 30 >= age >= 20:
        image = cv2.imread(image_file)
        lmk = lmks.loc[i, 'x0':'y67'].values.reshape([68, 2])
        draw_on_image(image, str(age), lmk)
        A20.append(image)
    elif age >= 85:
        image = cv2.imread(image_file)
        lmk = lmks.loc[i, 'x0':'y67'].values.reshape([68, 2])
        draw_on_image(image, str(age), lmk)
        A85.append(image)
A20 = np.array(A20)
A85 = np.array(A85)
A20 = A20[np.sort(np.random.choice(np.arange(len(A20)), len(A85), replace=False))]

video = cv2.VideoWriter('20vs85.avi', 0, 1, (400, 200))

for i1, i2 in zip(A20, A85):
    video.write(np.concatenate([i1, i2], axis=1))
video.release()