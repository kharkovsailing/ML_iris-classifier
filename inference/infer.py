import torch
import pandas as pd
import torch.nn as nn

class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def infer():
    df = pd.read_csv("data/iris_inference.csv")
    X = torch.tensor(df.values, dtype=torch.float32)

    model = IrisNet()
    model.load_state_dict(torch.load("iris_model.pt"))
    model.eval()

    predictions = model(X).argmax(dim=1).numpy()
    pd.DataFrame(predictions, columns=["prediction"]).to_csv("../inference_results.csv", index=False)

if __name__ == '__main__':
    infer()
