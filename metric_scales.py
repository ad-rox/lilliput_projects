from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # You can adjust the size of the model based on your computational resources
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def generate_text(prompt, max_length=150, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text based on the input prompt
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, num_return_sequences=1)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "The sun sets behind the mountains"
generated_poem = generate_text(prompt, max_length=100)
print("Generated Poem:")
print(generated_poem)
