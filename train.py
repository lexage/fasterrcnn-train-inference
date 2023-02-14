import torch
from datasets import PowerlinesDataset
from config import batch_size, num_workers, num_classes, \
                device, lr, momentum, weight_decay, epochs
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils import get_targets
from tqdm import tqdm
from numpy import mean


if __name__ == '__main__':
    train_set = PowerlinesDataset(train=True)
    val_set = PowerlinesDataset(train=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    data_loaders = {"train": train_loader, "val": val_loader}

    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    model.to(device);
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    cls_loss = []; box_loss = []; obj_loss = []; rpn_loss = []   
    cls_loss_val = []; box_loss_val = []; obj_loss_val = []; rpn_loss_val = []
    losses_dict = {'cls_loss': {"train" : cls_loss, "val": cls_loss_val}, 
                   'box_loss': {"train" : box_loss, "val": box_loss_val}, 
                   'obj_loss': {"train" : obj_loss, "val": obj_loss_val}, 
                   'rpn_loss': {"train" : rpn_loss, "val": rpn_loss_val}}
    model.train()
    for epoch in range(epochs):
        
        for phase in ["train", "val"]:
            
            loader = data_loaders[phase]
            
            for images, boxes in tqdm(loader):
                'forward pass'
                images, boxes = images.to(device), boxes.to(device)
                targets = get_targets(boxes)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, targets)

                    'saving losses'
                    outputs.keys()
                    for loss, key in zip(losses_dict.keys(), outputs.keys()):
                        losses_dict[loss][phase].append(outputs[key].item())
                        #print(f"{loss}: {outputs[key].item()}", end = ' ')

                    total_loss = sum(loss for loss in outputs.values())

                    'backward pass'
                    if phase == "train":
                        total_loss.backward()
                        optimizer.step()
                    
                optimizer.zero_grad()
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print("Train", *[f"{i}: {round(mean(losses_dict[i]['train'][epoch*len(train_loader):]), 4)}" \
            for i in losses_dict.keys()], sep = '  |  ')
        print("  Val", *[f"{i}: {round(mean(losses_dict[i]['val'][epoch*len(val_loader):]), 4)}" \
            for i in losses_dict.keys()], sep = '  |  ')
        #print(losses_dict)
    torch.save(model, 'weights/newest.pt')