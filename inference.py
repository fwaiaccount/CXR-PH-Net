import json
import os
import pydicom
import numpy as np
import skimage.transform
import argparse
import torch
from torchvision.transforms import Compose, ToTensor
from timm.models import create_model
from utils import ImgNor_6


def parse_args():
    parser = argparse.ArgumentParser(description='CXR-PH-Net Inference')
    parser.add_argument('--data-dir', metavar='DIR',
                        help='path to dataset (root dir)')
    parser.add_argument('--model-name', '-m', metavar='MODEL', default='vgg16',
                        help='model architecture (default: vgg16)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number classes in dataset')
    parser.add_argument('--model-path', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--use-gpu', default=False,
                        help='GPUS to use')
    parser.add_argument('--save-path', dest='save_path', required=False,
                        default='./CXR-PH-Net/result/test')
    return parser.parse_args()


class CXRPHNetModel:
    def __init__(self, use_gpu, model_name, num_class, model_path):
        self.use_gpu = use_gpu
        self.device = torch.device('cuda:0') if self.use_gpu else torch.device('cpu')
        self.model_name = model_name
        self.num_class = num_class
        self.model_path = model_path
        self.normalize = ImgNor_6
        self.transforms = Compose([ToTensor()])
        self.net = create_model(model_name=self.model_name, num_classes=self.num_class)
        self.net.load_state_dict(torch.load(self.model_path)['state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def preprocess(self, inp):
        """
        Args:
            inp: dcm_arr, (w,h)

        Returns: Tensor (*, 3, 1024,1024)
        """
        inp = skimage.transform.resize(inp, (1024, 1024))
        inp = np.concatenate((inp[...,np.newaxis],inp[...,np.newaxis],inp[...,np.newaxis]), axis=-1)
        inp = self.normalize(inp)
        inp = self.transforms(inp).float().cuda()
        return inp

    def predict(self, inp):
        pred_ = self.net(inp)
        pred_prob = torch.softmax(pred_, dim=1)
        pred = torch.argmax(pred_prob)
        return pred.cpu().numpy()

    @torch.no_grad()
    def run(self, img):
        data = self.preprocess(img)
        data = torch.unsqueeze(data, 0)
        pred = self.predict(data)
        return pred

def main():
    args = parse_args()
    instance = CXRPHNetModel(use_gpu=args.use_gpu, model_name=args.model_name, num_class=args.num_classes, model_path=args.model_path)
    data_l = os.listdir(args.data_dir)
    resutl_dict = {}
    for i in data_l:
        ds = pydicom.dcmread(os.path.join(args.data_dir, i))
        img_arr = np.array(ds.pixel_array)
        pred = instance.run(img_arr)
        resutl_dict[i] = pred
    with open(args.save_file+'.json', 'w') as f:
        json.dump(resutl_dict, f)

if __name__ == "__main__":
    main()


