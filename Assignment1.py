import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
from datetime import datetime

# Load pre-trained model and tokenizer
model_name = "gpt2-medium"  # Change this to any model you prefer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Manually set the padding token to be the end of sentence token
tokenizer.pad_token = tokenizer.eos_token

# Set the model to evaluation mode
model.eval()

def generate_sales_conversation(prompt, max_new_tokens=50):
    # Tokenize input text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            num_return_sequences=1, 
            temperature=0.9, 
            top_p=0.95, 
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            do_sample=True
        )
    
    # Decode and return the generated response
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example prompt for initiating a sales conversation
initial_prompt = "Salesman: Hello! Are you looking for a new laptop today?"
conversation = [("Salesman", initial_prompt.split(": ")[1], datetime.now().isoformat())]

# Generate conversation
for i in range(10):  # Let's generate a 10-turn conversation
    if i % 2 == 0:  # User's turn
        user_prompt = conversation[-1][1] + "\nUser:"
        user_response = generate_sales_conversation(user_prompt)
        conversation.append(("User", user_response.split(": ")[-1], datetime.now().isoformat()))
    else:  # Salesman's turn
        salesman_prompt = conversation[-1][1] + "\nSalesman:"
        salesman_response = generate_sales_conversation(salesman_prompt)
        conversation.append(("Salesman", salesman_response.split(": ")[-1], datetime.now().isoformat()))

# Convert conversation to DataFrame
conversation_df = pd.DataFrame(conversation, columns=["Speaker", "Text", "Timestamp"])

# Save to CSV
conversation_df.to_csv("sales_conversation.csv", index=False)

print("Conversation saved to sales_conversation.csv")
