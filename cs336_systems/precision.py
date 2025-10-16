import torch
from torch import nn

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
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for n, t in self.named_parameters():
            print("Within context parameter", n, "dtype", t.dtype)
        print("x dtype: ", x.dtype, ", shape: ", x.shape)
        fc1x = self.fc1(x)
        print("fc1x dtype:", fc1x.dtype)
        relux = self.relu(fc1x)
        print("relux:", relux.dtype)
        lnx = self.ln(relux)
        print("lnx dtype:", lnx.dtype)
        fc2x = self.fc2(lnx)
        print("fc2x dtype:", fc2x.dtype)
        return fc2x

if __name__ == "__main__":
    m = ToyModel(5, 5).to("cuda")
    inputs = torch.randn(30, 12, 5).to("cuda")
    labels = torch.randn(30, 12, 5).to("cuda")
    for n, x in m.named_parameters():
        print("Outside context parameter", n, "dtype", x.dtype)
        

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        y = m(inputs)
        loss = torch.mean((y - labels)**2)
        print("Loss dtype: ", loss.dtype)
    loss.backward()
    for n, x in m.named_parameters():
        print(n, "grad dtype: ", x.dtype)
