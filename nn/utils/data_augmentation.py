from skimage import exposure
import random


def img_aug_gen(gen, gamma=True, gamma_range=(0.7,1.3), random_flip=True,
                flip_ratio=0.5, hist_eq=True, hist_eq_ratio=0.1):
    while 1:
        img, label = next(gen)
        if random_flip & (random.random() > flip_ratio):
            img = img[:, ::-1]
            label = label[:, ::-1]

        if hist_eq & (random.random() < hist_eq_ratio):
            img = exposure.equalize_hist(img)

        elif gamma:
            img = exposure.adjust_gamma(img, random.uniform(gamma_range[0], gamma_range[1]))

        yield (img, label)

