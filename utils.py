import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms


def get_targets(boxes):
    ''''''
    'Function for casting boxes info into suitable for Faster-RCNN format'
    
    targets = []
    for i in range(len(boxes)): 
        d = {}
        d['boxes'] = boxes[i][1:][None, ...]
        d['labels'] = boxes[i][0][None, ...].to(torch.int64)
        targets.append(d)
    return targets

def visualize(dataset, i=5):
    ''''''''
    'Visualizes selected element from dataset'
    fig, ax = plt.subplots(1, figsize=(12,9))
    _, __ = dataset[i]
    ax.imshow(_.transpose(1,2,0))
    c, x1, y1, x2, y2 = __
    w = x2 - x1
    h = y2 - y1
    bbox = patches.Rectangle((x1,y1), w , h ,
                linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(bbox)
    plt.axis('off')
    print(__)
    plt.show()
    
def IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou    
      
       
def NMS(outputs, conf_thresh, iou_thresh):
    ''''''''
    'Input: list of dictionaries, each with "boxes", "scores", "labels"'
    'Output: list of filtered dictionaries'
    
    supressed_outputs = []
    if not isinstance(outputs, list) and isinstance(outputs, dict):
        outputs = [outputs]
    
    for output in outputs:
        
        assert set(output.keys()) == set(['boxes', 'scores', 'labels'])
        
        mask = output['scores'] >= conf_thresh
        
        boxes = output['boxes'][mask]
        scores = output['scores'][mask]
        labels = output['labels'][mask]
        
        filtered_idxs = nms(boxes, scores, iou_threshold=iou_thresh)
        
        boxes = boxes[filtered_idxs]
        scores = scores[filtered_idxs]
        labels = labels[filtered_idxs]
        supressed_outputs.append(
            {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
            }
        )
    
    return supressed_outputs
        
def save_result(img, result, img_path):
    
    my_dpi = 96 
    fig, ax = plt.subplots(1,figsize=(1980/my_dpi, 1080/my_dpi))
    ax.imshow(img.detach().cpu().numpy().transpose((1,2,0)))
    
    for box, label, score in zip(*result.values()):
        x1, y1, x2, y2 = box.detach().cpu().numpy()
        w = x2 - x1
        h = y2 - y1
        bbox = patches.Rectangle((x1,y1), w , h ,
                    linewidth=2, edgecolor='g', facecolor='none', label=f'{int(label)} | {round(score.item(), 2)}')
        ax.add_patch(bbox)
    plt.axis('off')
    plt.legend()
    fig.savefig(img_path, dpi=fig.dpi,bbox_inches='tight')
        

    