import torch
from torch import nn
import torch.cuda.nvtx as nvtx

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float32)
print("float32: ", s)

s = torch.tensor(0,dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print("float 16:", s)

s = torch.tensor(0,dtype=torch.bfloat16)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.bfloat16)
print("bfloat 16:", s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.float16)
print("float 32 acc, float 16 term:",s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01,dtype=torch.bfloat16)
print("float 32 acc, bfloat 16 term:",s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01,dtype=torch.float16)
    s += x.type(torch.float32)
print("float 32 acc, float 16 term, v2:",s)

s = torch.tensor(0.01, dtype=torch.bfloat16)
s = s.type(torch.float32)
print(s)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        inner_dim = in_features*4
        self.fc1 = nn.Linear(in_features, inner_dim, bias=False)
        self.ln = nn.LayerNorm(inner_dim)
        self.fc2 = nn.Linear(inner_dim, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for n, t in self.named_parameters():
            print("Within context parameter", n, "dtype", t.dtype)
        print("x dtype: ", x.dtype, ", shape: ", x.shape)
        with nvtx.range("fc1"):
            fc1x = self.fc1(x)        
            print("fc1x dtype:", fc1x.dtype)
        with nvtx.range("relu"):
            relux = self.relu(fc1x)
            print("relux:", relux.dtype)
        with nvtx.range("layernorm"):
            lnx = self.ln(relux)
            print("lnx dtype:", lnx.dtype)
        with nvtx.range("fc2"):
            fc2x = self.fc2(lnx)
            print("fc2x dtype:", fc2x.dtype)
        return fc2x

if __name__ == "__main__":
    m = ToyModel(2048, 2048).to("cuda")
    inputs = torch.randn(30, 12, 2048).to("cuda")
    labels = torch.randn(30, 12, 2048).to("cuda")
    for n, x in m.named_parameters():
        print("Outside context parameter", n, "dtype", x.dtype)
        

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with nvtx.range("forward"):
            y = m(inputs)
            loss = torch.mean((y - labels)**2)
            print("Loss dtype: ", loss.dtype)
    with nvtx.range("backward"):
        loss.backward()
    for n, x in m.named_parameters():
        print(n, "grad dtype: ", x.dtype)
