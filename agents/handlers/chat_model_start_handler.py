from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen

# Define a function that uses 'boxen' for styled printing
def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))

# Define a class for handling chat model events, inheriting from BaseCallbackHandler
class ChatModelStartHandler(BaseCallbackHandler):
    # Method triggered when the chat model starts processing
    def on_chat_model_start(self, serialized, messages, **kwargs):
        # Print a divider for clarity
        print("\n\n================== Sending Message ==============\n\n")

        # Iterate through each message in the conversation
        for message in messages[0]:
            # Different actions based on the type of message
            if message.type == "system":
                # Print system messages in a yellow box
                boxen_print(message.content, title=message.type, color="yellow")

            elif message.type == "human":
                # Print human messages in a green box
                boxen_print(message.content, title=message.type, color="green")

            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                # Special handling for AI messages that include a function call
                call = message.additional_kwargs['function_call']
                boxen_print(
                    f"Running tool {call['name']} with args {call['arguments']}", title=message.type, color="cyan")

            elif message.type == "ai":
                # Print AI messages in a blue box
                boxen_print(message.content, title=message.type, color="blue")

            elif message.type == "function":
                # Print function messages in a purple box
                boxen_print(message.content, title=message.type, color="purple")
            
            else:
                # Default styling for other types of messages
                boxen_print(message.content, title=message.type)
