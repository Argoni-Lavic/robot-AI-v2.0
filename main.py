print('starting')

import GPT_V1_0 as gpt

print('welcome')

while True:
    user_message = input("You: ")
    print(' ')
    if user_message.lower() == "exit":
        print('stoping')
        break
    response = gpt.send_message(user_message)
    print(f"Bot: {response}")
    print(' ')