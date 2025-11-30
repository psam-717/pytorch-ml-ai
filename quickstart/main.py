import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor() # converts downloaded PIL images to pytorch tensors with normalized values [0, 1]
)

# download training data from opening datasets
test_data = datasets.FashionMNIST( 
    root="data",
    train=False,
    download=True,
    transform=ToTensor()  
)

batch_size = 64

# create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N(batch size), C(no of channels), H (Height), W(Width)]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# check for a cuda available device else falls on the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # flattens the tensor from a 2D image ([N, 1, 28, 28]) to a 1D vector ([N, 784])
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # fully connected layer from 784 (28*28) inputs (flattened images) to 512 neurons 
            nn.ReLU(), # ReLU activation for non-linearity
            nn.Linear(512, 512), # second fully connected layer (512 to 512 neurons)
            nn.ReLU(), # another ReLU activation
            nn.Linear(512, 10) # Output layer mapping to 10 classes 
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)    


# Optimizing the Model Parameters
loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # optimizer

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # compute the prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward(),
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
# epochs is the sam as iteration
epoch = 5
for t in range(epoch):
    print(f"Epoch {t+1}\n--------------------------")
    train(train_dataloader, model,loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# save the model 
torch.save(model.state_dict(), "model.pth")
print("Saved Pytoch model state to model.pth")

model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')