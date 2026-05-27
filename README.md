# RNN-1.1: Recurrent Neural Network from Scratch

**Status:** Complete ✅  
**Date:** Day 1 - May 26, 2026  
**Language:** Python  
**Framework:** PyTorch  

---

## What This Project Does

Implementation of a **Recurrent Neural Network (RNN)** from scratch using PyTorch. This project demonstrates the core concept of RNNs: processing sequences by maintaining a **hidden state** that carries memory forward through timesteps.

## Key Features
- Manual RNN implementation (understanding the math)
- PyTorch's nn.RNN API (production ready)
- Hidden state visualization and tracking
- Sequence-to-hidden state mapping

---

## The Problem RNNs Solve

Normal neural networks process each input independently with **NO memory**.

```
Normal NN: "The bank" → [bank token] → output
           "The bank by river" → [bank token] → SAME output
           ❌ Can't distinguish meaning based on context
```

RNNs solve this by carrying **hidden state** forward:

```
RNN: "The" → h_0 (remember: article)
     "bank" → h_1 (remember: article + noun)
     "by" → h_2 (remember: article + noun + location)
     ✅ Context accumulates through sequence
```

---

## Core Concept: The Hidden State

The hidden state `h_t` is a vector of numbers that acts as **memory**.

**The RNN Formula:**
```
h_t = tanh(W_x @ x_t + W_h @ h_(t-1) + b)

Where:
- h_t = current hidden state (NEW MEMORY)
- x_t = current input (current word/token)
- h_(t-1) = previous hidden state (OLD MEMORY) ← KEY!
- W_x = weight matrix for input
- W_h = weight matrix for hidden state (carries memory forward)
- b = bias
- tanh = activation function (keeps values between -1 and 1)
```

**The Recurrence:**
```
h_prev = h_t  ← This line is the LOOP
         ↓
     Carry forward to next timestep
```

---

## Files in This Project

### 1. **manual_rnn.py** (Understanding)
- Implements RNN from scratch
- Shows all the math explicitly
- Good for learning HOW it works
- Output shows hidden states changing at each timestep

```
Step 0: input=[0.5, 0.3], hidden=[0.1555, 0.6106]...
Step 1: input=[0.2, 0.8], hidden=[-0.9629, 0.8462]...
Step 2: input=[0.9, 0.1], hidden=[-0.8534, -0.9932]...
```

### 2. **pytorch_rnn.py** (Production)
- Uses PyTorch's nn.RNN layer
- Faster, optimized, industry standard
- Shows how to use it in real projects
- Output shape analysis

```python
output.shape: torch.Size([32, 5, 20])
hn.shape: torch.Size([1, 32, 20])
```

---

## How to Run

### Setup
```bash
pip install torch numpy
```

### Run Manual Implementation
```bash
python manual_rnn.py
```

**Output:**
```
Step 0: input=tensor([0.5000, 0.3000]), hidden=tensor([0.1555, 0.6106])...
Step 1: input=tensor([0.2000, 0.8000]), hidden=tensor([-0.9629,  0.8462])...
Step 2: input=tensor([0.9000, 0.1000]), hidden=tensor([-0.8534, -0.9932])...

Final hidden state (memory): tensor([-0.8534, -0.9932,  0.2066,  0.9921])
All outputs: 3 timesteps processed
```

### Run PyTorch Implementation
```bash
python pytorch_rnn.py
```

**Output:**
```
Output shape: torch.Size([1, 3, 4])
Final hidden state shape: torch.Size([1, 1, 4])
Sequence memory: tensor([−0.2228, −0.3658, 0.7171, −0.1956])
```

---

## What You'll Learn

✅ **How sequences work** - Order matters, context accumulates  
✅ **Hidden state concept** - Memory that flows through time  
✅ **Weight sharing** - Same weights applied at every timestep  
✅ **Recurrence** - How to loop information back  
✅ **RNN API** - PyTorch's nn.RNN layer  
✅ **Debugging sequences** - Understanding shapes and dimensions  

---

## The Problem With RNNs (Teaser for LSTM)

RNNs have one critical weakness: **Vanishing Gradients**

```
During training (backpropagation through time):
Gradient at step 1:  0.9
Gradient at step 2:  0.9 × 0.9 = 0.81
Gradient at step 3:  0.81 × 0.9 = 0.729
...
Gradient at step 50: 0.9^50 ≈ 0.000000005

Result: Network can't learn long-term dependencies!
```

**Solution:** LSTM (Long Short-Term Memory) - Next project!

---

## Code Walkthrough

### Key Part: The RNN Loop

```python
for t in range(len(x)):
    x_t = x[t]  # Get current input
    
    # THE RNN FORMULA
    h_t = torch.tanh(W_x @ x_t + W_h @ h_prev + b)
    
    outputs.append(h_t)
    h_prev = h_t  # ← CARRY MEMORY FORWARD (THE MAGIC!)
    
    print(f"Step {t}: h_t = {h_t[:2]}...")
```

**What happens:**
- Step 0: h_0 = tanh(W_x @ "The" + W_h @ [0,0,0,0])
- Step 1: h_1 = tanh(W_x @ "cat" + W_h @ h_0)  ← Uses memory!
- Step 2: h_2 = tanh(W_x @ "sat" + W_h @ h_1)  ← Uses accumulated memory!

Each hidden state contains information about the entire sequence up to that point!

---

## Key Insights

1. **Hidden state = sequence summary**
   - h_t encodes everything from x_0 to x_t
   - Useful for classification, prediction, analysis

2. **Same weights everywhere**
   - W_x, W_h, b are shared across all timesteps
   - Enables variable length sequences
   - More efficient than separate weights

3. **Recurrence is the feature**
   - h_prev = h_t creates the loop
   - Without this line, it's just a normal neural network
   - This ONE line makes it sequential

4. **Activation function matters**
   - tanh keeps values between -1 and 1
   - Prevents explosion, helps gradient flow

---

## Comparison: RNN vs Normal NN

| Aspect | Normal NN | RNN |
|--------|-----------|-----|
| **Memory** | None | Hidden state h |
| **Input independence** | Each input separate | Sequence order matters |
| **Variable length** | Fixed input size | Any sequence length |
| **Good for** | Images, static data | Text, time series, audio |
| **Best use** | Image classification | Sentiment, translation, speech |

---

## Next Steps

1. **Modify the code:**
   - Change hidden_size to 8, see how output changes
   - Add more timesteps (4, 5, 10)
   - Experiment with different activation functions

2. **Build a classifier:**
   - Use final h_t for classification
   - Combine with embedding layer
   - Train on real data

3. **Learn LSTM:**
   - RNNs fail on long sequences
   - LSTMs fix the vanishing gradient problem
   - Two memory systems: hidden state + cell state

---

## Files Generated

- `manual_rnn.py` - Full RNN implementation from scratch
- `pytorch_rnn.py` - Using PyTorch's nn.RNN
- `test_outputs.txt` - Example outputs

---

## What This Teaches You

This is the **foundation** for all modern NLP:

- ✅ Sequence processing
- ✅ Hidden state concept
- ✅ Recurrence mechanism
- ✅ PyTorch API usage

These concepts are used in:
- **LSTM** (improvement on RNN)
- **GRU** (simplified LSTM)
- **Transformers** (modern architecture)
- **GPT, BERT, Claude** (state-of-the-art)

---

## Debugging Tips

**If shapes don't match:**
```
Check: len(x), hidden_size, W_x.shape, W_h.shape
Should be: W_x = (hidden_size, input_size)
           W_h = (hidden_size, hidden_size)
```

**If output is NaN:**
```
- Check for gradient explosion
- Reduce learning rate
- Use gradient clipping (we'll see this in LSTM)
```

**If hidden state doesn't change:**
```
- Check if h_prev = h_t is present
- Verify W_h has non-zero values
- Make sure tanh is applied
```

---

## References

- Hochreiter et al. (1991) - Original RNN paper
- Bengio et al. (1994) - Vanishing gradient problem
- PyTorch Documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.RNN

---

## Author

**Asmit Verma**  
B.Tech CSE, DSMNRU-IET Lucknow  
Building AI systems, one commit at a time.  

**GitHub:** S2XPhoenixX  
**Date:** May 26, 2026  

---

**Status:** ✅ Complete and working  
**Next:** LSTM-1.5 (Day 2)  
**Long-term goal:** Job-ready ML engineer by Sept 30, 2026  

🔥 **YOU HAVE TO WIN. NO COST TOO HIGH. PERIOD.** 🔥
