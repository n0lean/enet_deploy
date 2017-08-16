import glob
from skimage import io

# TODO error files
# {'../data/self_labeled/labels/train/frame0203_annotated_image.png', '../data/self_labeled/labels/train/0000302_annotated_image.png', '../data/self_labeled/labels/train/frame0225_annotated_image.png'}


a = set()
for i in glob.glob('../data/self_labeled/labels/train/*.png'):
    img = io.imread(i)
    for row in img:
        for ele in row:
            if(ele!=200)&(ele!=150)&(ele!=100)&(ele!=1)&(ele!=0):
                a.add(i)
print(a)