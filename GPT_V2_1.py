import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

class MemoryChatBot:
    def __init__(self, model_name="EleutherAI/gpt-neo-125M", memory_file="user_memory.json"):
        # Load GPT-Neo model
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  

        self.memory_file = memory_file
    
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
            json.dump(memory, file, indent=4)  
    
    def generate_input(self, user_message, length):
        """Generate input text including conversation history."""
        memory = self.load_memory()
        conversation_history = memory.get("conversation_history", [])[-length:]

        formatted_memory = "\n".join(
            [f"User: {turn['user']}\nBot: {turn['bot']}" for turn in conversation_history]
        )

        input_text = f"""{formatted_memory}

User: {user_message}
Bot:"""
        return input_text.strip()

    def generate_response(self, user_message):
        """Generate a response and prevent HTML-like outputs."""
        length = 10
        input_text = self.generate_input(user_message, length)

        max_length = 2048  

        while True:
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            token_length = len(inputs['input_ids'][0])

            if token_length <= max_length - 50:  
                break  
            else:
                length -= 1  
                if length <= 0:
                    raise ValueError("Conversation history is too long for token limit.")
                print(f"Warning: Input too long. Reducing memory length to {length} past messages.")
                input_text = self.generate_input(user_message, length)

        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_new_tokens=100,  
            temperature=0.5,  
            top_p=0.9,  
            repetition_penalty=2.2,
            do_sample=True,  
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Prevent HTML-like outputs
        if "<" in response or "input" in response or "form" in response:
            print("Warning: Model generated unexpected HTML content. Adjusting response...")
            response = "I'm sorry, I didn't understand that. Can you rephrase?"

        # Remove "Bot:" if it appears
        if "Bot:" in response:
            response = response.split("Bot:")[-1].strip()

        # Save conversation history
        memory = self.load_memory()
        conversation_history = memory.get("conversation_history", [])[-10:]
        conversation_history.append({"user": user_message, "bot": response})
        memory["conversation_history"] = conversation_history

        self.save_memory(memory)

        return response

# Usage
bot = MemoryChatBot()

def send_message(user_message):
    return bot.generate_response(user_message)
