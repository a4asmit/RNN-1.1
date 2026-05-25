import torch
import torch.nn as nn

x = torch.tensor([
    [0.5, 0.3],
    [0.2, 0.8],
    [0.9, 0.1]
])

rnn = nn.RNN(
    input_size=2,
    hidden_size=4,
    num_layers=1,
    batch_first=True
)

# Forward pass
# x needs shape (batch_size, seq_length, input_size)

X_batch = x.unsqueeze(0)

output, h_n = rnn(X_batch)

print(f"Output shape: {output.shape}")
print(f"Final hidden state: {h_n.shape}")

# output[0, -1, :] = final hidden state = summary of entire sequence
print(f"Sequence memory  {output[0, -1, :]}")

# For classification, use only the final hidden state
final_h = output[0, -1, :]  # (4,)
# Pass to linear layer for classification