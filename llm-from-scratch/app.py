import torch
import streamlit as st
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load the saved model parameters
loaded_variables = torch.load("model_parameters.pth")

# Extract each component
C = loaded_variables["C"]
W1 = loaded_variables["W1"]
b1 = loaded_variables["b1"]
W2 = loaded_variables["W2"]
b2 = loaded_variables["b2"]
block_size = loaded_variables["block_size"]
stoi = loaded_variables["stoi"]
itos = loaded_variables["itos"]

# Define the function to get probabilities of the next character
def get_next_char_probabilities(input_chars, C, W1, b1, W2, b2, block_size, stoi, itos):
    # Convert input characters to indices based on stoi (character-to-index mapping)
    context = [stoi.get(char, 0) for char in input_chars][-block_size:]  # Ensure context fits block size
    context = [0] * (block_size - len(context)) + context  # Pad if shorter than block size

    # Embedding the current context
    emb = C[torch.tensor([context])]
    
    # Pass through the network layers
    h = torch.tanh(emb.view(1, -1) @ W1 + b1)
    logits = h @ W2 + b2
    
    # Compute the probabilities
    probs = F.softmax(logits, dim=1).squeeze()  # Squeeze to remove unnecessary dimensions
    
    # Return probabilities for each character
    return {itos[i]: probs[i].item() for i in range(len(probs))}

# Define the function to generate a new Pokémon name based on input characters
def generate_name(input_chars, C, W1, b1, W2, b2, block_size, stoi, itos, g=torch.Generator()):
    # Initialize the context with input characters
    context = [stoi.get(char, 0) for char in input_chars][-block_size:]  # Ensure context fits block size
    context = [0] * (block_size - len(context)) + context  # Pad if shorter than block size

    # Generate name character by character
    out = []
    while True:
        # Get embedding and pass through network
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        
        # Sample from probabilities
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        
        # Update context with new character
        context = context[1:] + [ix]
        
        # Append character to output name
        out.append(ix)
        
        # Break if end token is reached
        if ix == 0:
            break
    
    # Convert indices to characters
    generated_name = ''.join(itos[i] for i in out if i in itos)
    return generated_name

# Streamlit app
st.title("Pokémon Name Generator")
st.write("Enter a sequence of characters, and see the model's predictions for the next character or generate a new Pokémon name!")

# Input field
input_chars = st.text_input("Enter starting characters:", value="bul")

# Get probabilities and generate name if input is provided
if input_chars:
    # Step 1: Display Next Character Probabilities
    next_char_probs = get_next_char_probabilities(input_chars, C, W1, b1, W2, b2, block_size, stoi, itos)
    
    st.subheader(f"Predicted probabilities for the next character after '{input_chars}'")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 3)
    ax.bar(next_char_probs.keys(), next_char_probs.values(), color="skyblue")
    ax.set_xlabel("Next Character")
    ax.set_ylabel("Probability")
    ax.set_title("Next Character Probabilities")
    st.pyplot(fig)
    
    # Step 2: Generate New Pokémon Name
    st.subheader("Generated Pokémon Name")
    generated_name = generate_name(input_chars, C, W1, b1, W2, b2, block_size, stoi, itos)
    st.write(f"Generated Name: **{input_chars}{generated_name}**")
