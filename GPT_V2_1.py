import json
from transformers import GPTNeoForCausalLM, AutoTokenizer

MEM_PROMPT = '(Based on the context above, do you recall anything relevant from memory that may assist in generating a helpful response?)'

class MemoryChatBot:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.7B", memory_file="user_memory.json"):
        # Load the GPT-Neo 1.7B model and tokenizer
        model = GPTNeoForCausalLM.from_pretrained(model_name, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Assign a padding token (GPT-Neo does not have one by default)
        self.tokenizer.pad_token = self.tokenizer.eos_token  

        self.memory_file = memory_file
    
    def create_input_for_gpt(self, context):
        """Formats conversation history so GPT-Neo understands it properly."""
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
            json.dump(memory, file, indent=4)  # Added indent for readability
    
    def generate_input(self, user_message, length):
        """Generate input text by including conversation history up to a specific length."""
        # Load memory
        memory = self.load_memory()

        # Keep only the last `length` messages to prevent excessive input length
        conversation_history = memory.get("conversation_history", [])[-length:]

        # Format the stored memory
        memory_text = "### STORED MEMORY ###\n"
        for i, turn in enumerate(conversation_history):
            memory_text += f"{i+1}. User: {turn['user']}\n   Bot: {turn['bot']}\n"

        # Modify prompt for clearer memory usage
        input_text = f"""{memory_text}

### CURRENT CONVERSATION ###
User: {user_message}
Bot: {MEM_PROMPT}"""
        return input_text

    def generate_response(self, user_message):
        """Generate a response based on memory, with a clearer prompt for GPT-Neo to 'remember'."""

        # Initialize the length of stored memory
        length = 10

        # Generate initial input text
        input_text = self.generate_input(user_message, length)

        # Ensure input length is within token limit
        while True:
            # Tokenize the input to check its length
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=False)
            token_length = len(inputs['input_ids'][0])  # Check the number of tokens

            if token_length <= 2048:  # Adjust for GPT-Neo's token limit (2048)
                break  # Exit the loop if the input fits within the token limit
            else:
                length -= 1  # Reduce the number of conversation history messages
                if length <= 0:
                    raise ValueError("The conversation history is too long to fit within the token limit.")
                print(f"Warning: Input too long. Reducing stored memory length to {length} past messages.")
                print(' ')
                input_text = self.generate_input(user_message, length)  # Regenerate input with reduced length

        # Tokenize final input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)

        # Generate response
        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,  
            temperature=0.1,  # Low randomness to improve recall accuracy
            repetition_penalty=1.8,  # Reduces repetition and irrelevant answers
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Extract only the bot's latest response
        if "Bot:" in response:
            response = response.split("Bot:")[-1].strip()

        # Save updated memory
        memory = self.load_memory()
        conversation_history = memory.get("conversation_history", [])[-10:]
        conversation_history.append({"user": user_message, "bot": response})
        memory["conversation_history"] = conversation_history

        self.save_memory(memory)

        return response

# Usage example
bot = MemoryChatBot()

def send_message(user_message):
    response = bot.generate_response(user_message)[len(MEM_PROMPT):]

    return response
