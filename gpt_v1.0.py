import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class MemoryChatBot:
    def __init__(self, model_name="gpt2", memory_file="user_memory.json"):
        # Load the GPT-2 model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Assign a padding token (GPT-2 does not have one by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token  

        self.memory_file = memory_file
    
    def create_input_for_gpt(self, context):
        """Formats conversation history so GPT-2 understands it properly."""
        context_str = "\n".join([f"User: {turn['user']}\nBot: {turn.get('bot', '')}" for turn in context])
        return context_str.strip() + "\nBot:"  # Ensure model knows where to generate response

    def load_memory(self):
        """Load or initialize memory from a file."""
        try:
            with open(self.memory_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {"conversation_history": []}

    def save_memory(self, memory):
        """Save the memory to the file."""
        with open(self.memory_file, "w") as file:
            json.dump(memory, file)

    def generate_response(self, user_message):
        """Generate a response based on conversation history."""
        # Retrieve memory
        memory = self.load_memory()

        # Keep only the last 5 messages to prevent excessive input length
        context = memory.get("conversation_history", [])[-5:]
        context.append({"user": user_message})

        # Prepare input for GPT-2
        input_text = self.create_input_for_gpt(context)

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

        # Ensure input is valid
        if inputs['input_ids'].size(1) == 0:
            raise ValueError("Tokenized input is empty. Check input_text or tokenizer settings.")

        attention_mask = inputs['attention_mask']

        # Generate response
        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=attention_mask,
            max_new_tokens=50,  # Prevents extremely long responses
            temperature=0.7,  # Increases randomness
            repetition_penalty=1.2,  # Reduces repeating responses
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Extract only the bot's latest response
        if "Bot:" in response:
            response = response.split("Bot:")[-1].strip()

        return response

# Usage example
bot = MemoryChatBot()

def send_message(user_message):
    response = bot.generate_response(user_message)
    return response
