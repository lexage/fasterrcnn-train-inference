from utils import NMS
import argparse
from datasets import PowerlinesDataset
import torch
from utils import NMS
import os
from utils import save_result
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Custom Faster R-CNN detector')
    parser.add_argument('data', metavar='DIR', nargs='?', default='data/images/test',
                        help='path to dataset (default: data/images/test)')
    parser.add_argument('-i', '--iou-thresh', metavar='IOU', default=0.25,
                        help='iou-thresh for non-maximum supression, default: 0.25')
    parser.add_argument('-c', '--conf-thresh', default=0.5, metavar='CONF',
                    help='conf-thresh for non-maximum supression, default: 0.5')
    parser.add_argument('-w', '--weights', default='weights/new.pt', metavar='WEIGHTS',
                    help='relative path to model weights, default: weights/new.pt')
    parser.add_argument('-d', '--device', default='cpu', metavar='DEV',
                    help='device, default: cpu')
    parser.add_argument('-r', '--results', default='results', metavar='RES',
                    help='results directory, default: results')
    args = parser.parse_args()
    
    print('Starting...')
    
    data_path = args.data
    iou_thresh = args.iou_thresh
    conf_thresh = args.conf_thresh
    weights = args.weights
    device = args.device
    results_path = args.results
    
    print('loading data')
    dataset = PowerlinesDataset(train=False, data_path=data_path, test=True)
    device = 'cpu'
    model = torch.load(weights)
    model.to(device)
    model.eval()
    
    exp_num = 1
    while os.path.exists(f"{results_path}/exp{exp_num}"):
        exp_num += 1
    os.mkdir(f"{results_path}/exp{exp_num}/")
    
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        img = img.to(device)
        result = model(img)
        result = NMS(result, conf_thresh=conf_thresh, iou_thresh=iou_thresh)[0]
        
        basename = dataset.getname(i)
        img_path = f"{results_path}/exp{exp_num}/{basename}"
        save_result(img=img[0], result=result, img_path=img_path)
    print("detection's completed!")        

