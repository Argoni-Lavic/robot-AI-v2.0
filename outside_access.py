import requests

# Replace <ngrok-url> with the URL given by ngrok
url = input('input the url for colab conection: ') + '/generate'

def send_message(input_message):
    data = {
        "input": input_message
    }
    try:
        response = requests.post(url, json=data)
        print(response.status_code)  # Print status code to see if the request was successful
        if response.status_code == 200:
            try:
                response_json = response.json()
                print(response_json)  # Print response JSON
            except requests.exceptions.JSONDecodeError:
                print("Error: Response is not valid JSON")
                print("Response Text:", response.text)  # Print raw response
        else:
            print(f"Error: Received status code {response.status_code}")
    except Exception as e:
        print("Error:", e)
