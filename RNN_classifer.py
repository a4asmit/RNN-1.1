import torch 
import torch.nn as nn
import torch.optim as optim 

class SimpleRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, h_n = self.rnn(x)
        h_final = output[:, -1, :]
        logits = self.fc(h_final)
        return logits
    
model = SimpleRNNClassifier(vocab_size=1000, embed_dim=50, hidden_size=60, num_classes=2)
optimiser = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Dummy input
x = torch.randint(0, 1000, (32, 20))  # batch=32, seq_length=20
y = torch.randint(0, 2, (32,))  # labels

logits = model(x)
loss = loss_fn(logits, y)

optimiser.zero_grad()
loss.backward()
optimiser.step()

print(f"Loss: {loss.item}")