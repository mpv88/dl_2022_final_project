import torch

print('_________testing if GPU is available_________')

# check torch + cuda versions installed
assert torch.__version__ == '1.13.1+cu117'

# does PyTorch see any GPU?
assert torch.cuda.is_available() == True

# how many GPUs does it see?
assert torch.cuda.device_count() == 1 

# which is the current available device number?
assert torch.cuda.current_device() == 0

# get details (address, name) on available device number?
# print(torch.cuda.device(0)) memory address
assert torch.cuda.get_device_name(0) == 'NVIDIA GeForce GTX 850M'

#Set up sample tensor on GPU:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.rand(3, 3).to(device)

# is x tensor a GPU tensor?
assert x.is_cuda == True

# is x tensor stored on cuda:=0?
assert x.device == torch.device('cuda', 0) # default is torch.device('cpu', 0), else torch.device('cuda', 0) if gpu enabled
'''
# set default tensor type to CUDA:
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# is this model stored on the GPU?
all(p.is_cuda for p in my_model.parameters())
'''
print('_________test successfully completed!_________')