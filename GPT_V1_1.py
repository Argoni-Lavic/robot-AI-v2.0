import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MemoryChatBot:
    def __init__(self, model_name="EleutherAI/gpt-Neo-2.7B", memory_file="user_memory.json"):
        # Load GPT-NeoX model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

        # Assign a padding token if missing
        self.tokenizer.pad_token = self.tokenizer.eos_token  

        self.memory_file = memory_file
    
    def create_input_for_gpt(self, context):
        context_str = "\n".join([f"User: {turn['user']}\nBot: {turn.get('bot', '')}" for turn in context])
        return context_str.strip() 

    def load_memory(self):
        try:
            with open(self.memory_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            return {"conversation_history": []}

    def save_memory(self, memory):
        with open(self.memory_file, "w") as file:
            json.dump(memory, file)

    def generate_response(self, user_message):
        memory = self.load_memory()
        context = memory.get("conversation_history", [])
        context.append({"user": user_message})

        input_text = self.create_input_for_gpt(context)
        if not input_text.strip():
            input_text = "Hello! How can I help you today?"

        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")

        # Generate response
        outputs = self.model.generate(
            inputs['input_ids'], 
            attention_mask=inputs['attention_mask'],
            max_new_tokens=256,  
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Update memory
        memory["conversation_history"].append({"user": user_message, "bot": response})
        self.save_memory(memory)

        return response

# Usage example
bot = MemoryChatBot()

def send_message(user_message):
    response = bot.generate_response(user_message)
    return response
