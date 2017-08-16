import pickle
import numpy as np

class self_labeled(object):
    def __init__(self, binary_dict=False):
        """
        trainId  category     prev_id
        0        object       100
        1        road         1
        2        person       150
        3        sky          200
        4        unlabeled    0
        """
        if binary_dict:
            self.dictionary = {
                100: 0,
                1: 1,
                150: 0,
                200: 0,
                0: 0
            }
        else:
            self.dictionary = {
                100: 0,
                1: 1,
                150: 2,
                200: 3,
                0: 4
            }



    def label_trans(self, idx):
        return self.dictionary[idx]

    def img_label_trans2(self, img):
        # not changed to one hot
        w, h = img.shape
        arr = np.zeros((w, h))
        for row_idx, row in enumerate(img, 0):
            for ele_idx, ele in enumerate(row, 0):
                arr[row_idx, ele_idx] = self.label_trans(int(ele))
        return arr

    def label2one_hot(self, img, dimension=5):
        w, h = img.shape
        arr = np.zeros((w, h, dimension))
        for row_idx, row in enumerate(img, 0):
            for ele_idx, ele in enumerate(row, 0):
                arr[row_idx, ele_idx, ele] = 1
        return arr

    def img_label_trans(self, img, dimension=5):
        # img should be a 2d array.
        w, h = img.shape
        arr = np.zeros((w,h,dimension))
        for row_idx, row in enumerate(img, 0):
            for ele_idx, ele in enumerate(row, 0):
                arr[row_idx, ele_idx, self.label_trans(int(ele))] = 1
        return arr


class cityscape2mine(object):
    def __init__(self):
        """
        trainId  category
        0        object 
        1        road
        2        person
        3        sky
        4        unlabeled
        
        pre_trainId  post_trainId   name 
        255          4              
        0            1              road
        1            1              sidewalk
        2,3,4,5      0              building, wall, fence, pole
        6,7          0              traffic light
        8,9          4              nature (vegetation, terrain)
        10           3              sky
        11,12        2              human (person, rider)
        13,14,15,16  0              vehicle (car, truck, bus, train)
        17,18        0              motorcycle, bicycle
        -1           4              license plate

        """
        self.dictionary = {
            255: 4,
            0: 1,
            1: 1,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 4,
            9: 4,
            10: 3,
            11: 2,
            12: 2,
            13: 0,
            14: 0,
            15: 0,
            16: 0,
            17: 0,
            18: 0,
            -1: 4,
        }

        self.id2trainid_dict = {
            0: 255,
            1: 255,
            2: 255,
            3: 255,
            4: 255,
            5: 255,
            6: 255,
            7: 0,
            8: 1,
            9: 255,
            10: 255,
            11: 2,
            12: 3,
            13: 4,
            14: 255,
            15: 255,
            16: 255,
            17: 5,
            18: 255,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: 255,
            30: 255,
            31: 16,
            32: 17,
            33: 18,
            -1: -1
        }

    def id2tranid(self, idx):
        return self.id2trainid_dict[idx]

    def trainid2mine(self, idx):
        return self.dictionary[idx]

    def id2mine(self, idx):
        return self.dictionary[self.id2trainid_dict[idx]]

    def img_label_trans(self, img, dimension=5):
        # img should be a 2d array.
        w, h = img.shape
        arr = np.zeros((w,h,dimension))
        for row_idx, row in enumerate(img, 0):
            for ele_idx, ele in enumerate(row, 0):
                arr[row_idx, ele_idx, self.id2mine(int(ele))] = 1
        return arr

if __name__ == '__main__':
    test = cityscape2mine()
    img = np.zeros((10,10))
    img[3,4] = 1
    print(test.img_label_trans(img, 5))