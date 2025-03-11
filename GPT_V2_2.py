import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

class MemoryChatBot:
    def __init__(self, model_name="EleutherAI/gpt-neo-1.3B", memory_file="user_memory.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPTNeoForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token  
        self.memory_file = memory_file

    def load_memory(self):
        try:
            with open(self.memory_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {"conversation_history": []}

    def save_memory(self, memory):
        with open(self.memory_file, "w") as file:
            json.dump(memory, file, indent=4)

    def generate_input(self, user_message, length):
        """Format conversation history properly."""
        memory = self.load_memory()
        conversation_history = memory.get("conversation_history", [])[-length:]

        formatted_memory = "\n".join(
            [f"User: {turn['user']}\nBot: {turn['bot']}" for turn in conversation_history]
        )

        # Ensure proper separation
        input_text = f"{formatted_memory}\n\nUser: {user_message}\nBot:"
        return input_text.strip()

    def generate_response(self, user_message):
        length = 10
        input_text = self.generate_input(user_message, length)
        max_length = 2048  

        while True:
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            token_length = len(inputs['input_ids'][0])

            if token_length <= max_length - 50:
                break  
            else:
                length -= 1
                if length <= 0:
                    raise ValueError("Conversation history is too long for token limit.")
                print(f"Warning: Input too long. Reducing memory length to {length} past messages.")
                input_text = self.generate_input(user_message, length)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100,  
                temperature=0.2,  # Lowered for better control
                top_p=0.95,
                repetition_penalty=2.8,  # Increased to avoid loops
                do_sample=True,  
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Prevent repeated user input
        response = response.split("\nUser:")[0].strip()

        # Prevent unwanted HTML/code responses
        if "<" in response or "public static" in response or "String" in response:
            print("Warning: Model generated unwanted content. Adjusting response...")
            response = "I'm not sure about that. Can you rephrase?"

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
