import torch
import torch.nn as nn

x = torch.tensor([
    [0.5, 0.3], #Embedding of "The"
    [0.2, 0.8], #Embedding of "cat"
    [0.9, 0.1]  #Embedding of "sat"
])  #shape (3,2) #3 timesteps, 2 features each

#RNN paramaters

input_size = 2
hidden_size = 4
W_x = torch.randn(hidden_size, input_size)
W_h = torch.randn(hidden_size, hidden_size)
b = torch.randn(hidden_size)

#initial hidden stages

h_prev = torch.zeros(hidden_size)

#Process sequences 

outputs = []
for t in range(len(x)):
    x_t = x[t]

    #RNN formula 
    h_t = torch.tanh(W_x @ x_t + W_h @h_prev + b )

    outputs.append(h_t)
    h_prev = h_t #carry forward 

    print(f" Step {t}: input={x_t}, hidden={h_t[0:2]}...")

print(f"\nFinal hidden state (memory): {h_prev}")
print(f"All outputs: {len(outputs)} timesteps processed")