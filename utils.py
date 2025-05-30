import numpy as np
from torch.utils.data import Dataset
import pydicom
import skimage
def ImgNor_1(x):
    x = x/np.max(x)*255.0
    return x

def ImgNor_6(x):
    lower = np.percentile(x, 0.02)
    upper = np.percentile(x, 99.8)
    x[x < lower] = lower
    x[x > upper] = upper
    x -= x.mean()
    x /= x.std()
    return x

def load_img_label_list(fpath):
    with open(fpath) as f:
        lines = f.readlines()
    data_l = [[line.split(' ')[0], line.split(' ')[1].strip()] for line in lines]
    return data_l

class ChestXrayDataset(Dataset):
    def __init__(self, data_file, phase='train', normalization=None, transform=None, target_transform=None):
        super(ChestXrayDataset, self).__init__()
        data_l = load_img_label_list(data_file)
        self.data = data_l
        self.transform = transform
        self.target_transform = target_transform
        self.normalization = normalization

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        ds = pydicom.dcmread(img_path)
        img_arr = np.array(ds.pixel_array, dtype='int')
        img_arr = skimage.transform.resize(img_arr,(1024, 1024))
        img_arr = np.concatenate((img_arr[...,np.newaxis], img_arr[...,np.newaxis], img_arr[...,np.newaxis]), axis=-1)

        if self.normalization:
            img_arr = self.normalization(img_arr)
        if self.transform:
            img_arr = self.transform(img_arr)
        if self.target_transform:
            target = self.target_transform(target)
        return img_arr, int(target)

    def filename(self, index):
        return self.data[index]
