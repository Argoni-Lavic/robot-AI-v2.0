import GPT_V1_1 as gpt

while True:
    user_message = input("You: ")
    if user_message.lower() == "exit":
        break
    response = gpt.send_message(user_message)
    print(f"Bot: {response}")