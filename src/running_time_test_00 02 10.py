# Run in 14' on HP ENVY 17 (Intel Core i7-4510U CPU @2.00GHz×4, 8GB RAM, 1TB HDD) with NVIDIA Corporation GM107M [GeForce GTX 850M] 
# torch.cuda.OutOfMemoryError: CUDA out of memory. 
# Tried to allocate 180.00 MiB (GPU 0; 3.95 GiB total capacity; 3.21 GiB already allocated; 104.12 MiB free; 3.37 GiB reserved in total by PyTorch) 
# If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  
# See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF


import torch
import torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(torch.nn.Module):
  def __init__(self, dimensions):
    super().__init__()
    self.layers = torch.nn.ModuleList([ReLULayer(dimensions[i], dimensions[i+1]) for i in range(len(dimensions)-1)])
  
  def predict(self, x):
    goodness_per_label = []
    for label in range(10):
      x_lab = label_images(x, label)
      goodness = []
      for i, layer in enumerate(self.layers):
        x_lab = layer(x_lab)
        if i > 0:
          goodness.append(pow(x_lab, 2).mean(dim=1, keepdim=False)) # Σ[relu(h)^2]
      goodness_per_label.append(sum(goodness).unsqueeze(1))
    goodness_per_label = torch.cat(goodness_per_label, 1)
    return torch.argmin(goodness_per_label, dim=1) # set to argmin for point 3.B
    
  def train(self, x_pos, x_neg):
    for layer in self.layers:
      x_pos, x_neg = layer.train(x_pos, x_neg)

def normalize(x):
  return torch.nn.functional.normalize(x, p=2.0, dim=1, eps=1e-12) #x/x.norm(p=2, dim=1, keepdim=True, dtype=torch.float) activation normalized by L2-norm to get direction only

class ReLULayer(torch.nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.linear = torch.nn.Linear(in_features, out_features)
    self.relu = torch.nn.ReLU() # ['torch.nn.ReLU()', 'torch.nn.LeakyReLU(0.25)', 'torch.nn.ELU(alpha=1.0)', 'torch.nn.CELU(alpha=1.0)', 'torch.nn.SELU()', 'torch.nn.GELU(approximate='none')', 'torch.nn.Threshold(threshold=0, value=-.5)', 'torch.nn.Sigmoid()', 'torch.nn.Tanh()', 'torch.nn.Softmax(dim=1)']
    self.optimizer = torch.optim.Adam(self.parameters(), lr=0.03)
    self.threshold = 2 # set to 0.5 for point 3.A
    self.num_epochs = 1000 # set to 2000 for point 3.C
    
  def forward(self, x):
    x_direction = normalize(x)
    return self.relu(self.linear(x_direction))

  def train(self, x_pos, x_neg):
    for i in range(self.num_epochs):
      #positive_goodness = torch.zeros(x_pos.size(dim=0)).to(device) # uncomment for point 3.C
      #negative_goodness = torch.zeros(x_neg.size(dim=0)).to(device) # uncomment for point 3.C
      positive_goodness = self.forward(x_pos).pow(2).mean(dim=1, keepdim=False) # positive goodness step
      negative_goodness = self.forward(x_neg).pow(2).mean(dim=1, keepdim=False) # negative goodness step
      l = torch.log(1 + torch.exp(torch.cat([+positive_goodness - self.threshold, # maximize positive goodness            [-negative_goodness + self.threshold,
                                             -negative_goodness + self.threshold]))).mean() # minimize negative goodness   positive_goodness - self.threshold]))).mean()
      self.optimizer.zero_grad()
      l.backward(retain_graph=False)
      self.optimizer.step()
    return self.forward(x_pos).detach(), self.forward(x_neg).detach()

# instantiate neural net and fix seed
net = Net([784, 500, 500]).to(device)
torch.manual_seed(0)

# import normalized dataset to dataloaders
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,)), torchvision.transforms.Lambda(torch.flatten)])

trainset = torchvision.datasets.MNIST('./data/', transform=transform,  train=True, download=True)
testset = torchvision.datasets.MNIST('./data/', transform=transform, train=False, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)

def label_images(images, labels):
  '''input are: (batch_size, 28x28) images and (batch_size, 1) labels 
     output is: (batch_size, 28x28) label-embedded images'''
  embedded_images = images.detach().clone() # in our case (60000x784)
  embedded_images[[ _ for _ in range(embedded_images.size(dim=0))], labels] = images.max(dim=1)[0]
  plt.imshow(embedded_images[1234].cpu().reshape(28,28), cmap='gray') # print sample embedded image
  return embedded_images

# unpack train set and create positive/negative samples 
x, y = next(iter(trainloader))
x = x.to(device)
y = y.to(device)

x_pos = label_images(x, y)
rnd = torch.randperm(x.size(0))
x_neg = label_images(x, y[rnd])

# launch training
net.train(x_pos, x_neg)
print('Train accuracy:', net.predict(x).eq(y).float().mean().item())

# unpack test set
x_test, y_test = next(iter(testloader))
x_test = x_test.to(device)
y_test = y_test.to(device)

# launch trained model on test set
print('Test accuracy:', net.predict(x_test).eq(y_test).float().mean().item())
#0.9298833608627319
#0.9292999505996704