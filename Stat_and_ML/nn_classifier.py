import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# モデルの定義
class NN_classifier(nn.Module):
    def __init__(self, X, Y):
        super(NN_classifier, self).__init__()

        self.fc1 = nn.Linear(X.shape[1], X.shape[0])
        self.fc2 = nn.Linear(len(Y), 1)

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

        self.train_data = TensorDataset(X_tensor, Y_tensor)
        self.train_loader = DataLoader(self.train_data, batch_size=2, shuffle=True)

        #X_train, X_test, Y_train, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.2, random_state=42)

    def fit(self, model):
        # 損失関数とオプティマイザー
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 学習
        epochs = 100
        for epoch in range(epochs):
            for X_batch, Y_batch in self.train_loader:
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                # 損失の計算
                loss = criterion(outputs, Y_batch.view(-1, 1))
                # 逆伝播
                loss.backward()
                
                optimizer.step()

            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
    def validate(self, X_test, Y_test, model):
        X_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_test.values, dtype=torch.float32)

        test_data = TensorDataset(X_tensor, Y_tensor)
        test_loader = DataLoader(test_data, batch_size=1)

        model.eval()

        total = 0
        correct = 0

        # 勾配計算を無効化
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                outputs = model(X_batch)
                # 出力を0または1に丸める
                predicted = (outputs.data > 0.5).float()
                total += Y_batch.size(0)
                correct += (predicted.view(-1) == Y_batch).sum().item()

        # 精度の計算
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')



def main():
    data = {
    'A': [12.3, 15.6, 17.8, 13.4, 16.7],
    'B': [11.7, 13.4, 18.8, 12.3, 14.6],
    'Y': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    X = df[['A', 'B']]
    Y = df['Y']

    nn_model = NN_classifier(X, Y)
    nn_model.fit(nn_model)
    
    nn_model.validate(X, Y, nn_model)

if __name__ == '__main__':
    main()