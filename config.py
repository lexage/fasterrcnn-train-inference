import torch.cuda 


batch_size = 16
num_workers = 2
num_classes = 3
lr = 0.001
momentum = 0.9
weight_decay = 5e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 20