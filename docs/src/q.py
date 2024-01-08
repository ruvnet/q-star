#                 - Q* Agents
#        /\__/\   - q.py
#       ( o.o  )  - v0.0.1
#         >^<     - by @rUv

import os
import autogen
from autogen import config_list_from_json, UserProxyAgent, AssistantAgent, GroupChatManager, GroupChat
import numpy as np
import random
import logging
import threading
import sys
import time

# Determine the directory of the script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Set up logging to capture errors in an error_log.txt file, stored in the script's directory
log_file = os.path.join(script_directory, 'error_log.txt')
logging.basicConfig(filename=log_file, level=logging.ERROR)

# Check if running in Replit environment
if 'REPL_ID' in os.environ:
    print("Running in a Replit environment. Adjusting file paths accordingly.")
    # You may need to adjust other paths or settings specific to the Replit environment here
else:
    print("Running in a non-Replit environment.")
# Define the Q-learning agent class
class QLearningAgent:
    # Initialization of the Q-learning agent with states, actions, and learning parameters
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.95):
        self.states = states
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # Initialize Q-table with zeros
        self.q_table = np.zeros((states, actions))

    # Choose an action based on the exploration rate and the Q-table
    def choose_action(self, state, exploration_rate):
        if random.uniform(0, 1) < exploration_rate:
            # Explore: choose a random action
            return random.randint(0, self.actions - 1)
        else:
            # Exploit: choose the best action based on the Q-table
            return np.argmax(self.q_table[state, :])

    # Update the Q-table based on the agent's experience (state, action, reward, next_state)
    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (target - predict)

# ASCII Loading Animation Frames
frames = ["[■□□□□□□□□□]", "[■■□□□□□□□□]", "[■■■□□□□□□□]", "[■■■■□□□□□□]",
          "[■■■■■□□□□□]", "[■■■■■■□□□□]", "[■■■■■■■□□□]", "[■■■■■■■■□□]",
          "[■■■■■■■■■□]", "[■■■■■■■■■■]"]

# Global flag to control the animation loop
stop_animation = False

# Function to animate the loading process continuously
def animate_loading():
    global stop_animation
    current_frame = 0
    while not stop_animation:
        sys.stdout.write('\r' + frames[current_frame])
        sys.stdout.flush()
        time.sleep(0.2)
        current_frame = (current_frame + 1) % len(frames)
    # Clear the animation after the loop ends
    sys.stdout.write('\r' + ' ' * len(frames[current_frame]) + '\r')
    sys.stdout.flush()

# Function to start the loading animation in a separate thread
def start_loading_animation():
    global stop_animation
    stop_animation = False
    t = threading.Thread(target=animate_loading)
    t.start()
    return t

# Function to stop the loading animation
def stop_loading_animation(thread):
    global stop_animation
    stop_animation = True
    thread.join()  # Wait for the animation thread to finish
    # Clear the animation after the thread ends
    sys.stdout.write('\r' + ' ' * len(frames[-1]) + '\r')
    sys.stdout.flush()

# Load the AutoGen configuration from a JSON file
try:
    config_list_gpt4 = config_list_from_json("OAI_CONFIG_LIST.json")
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    print(f"Failed to load configuration: {e}")
    sys.exit(1)

llm_config = {"config_list": config_list_gpt4, "cache_seed": 42}

# Create user and assistant agents for the AutoGen framework
user_proxy = UserProxyAgent(name="User_proxy", system_message="A human admin.", code_execution_config={"last_n_messages": 3, "work_dir": "./tmp"}, human_input_mode="NEVER")
coder = AssistantAgent(name="Coder", llm_config=llm_config)
critic = AssistantAgent(name="Critic", system_message="Critic agent's system message here...", llm_config=llm_config)

# Set up a group chat with the created agents
groupchat = GroupChat(agents=[user_proxy, coder, critic], messages=[], max_round=20)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# Print initial instructions
# ASCII art for "Q*"
print("  ____  ")
print(" / __ \\ ")
print("| |  | |")
print("| |__| |")
print(" \____\ ")
print("       * Created by @rUv")
print("  ")
print("Welcome to the Q-Star  Agent, powered by the Q* algorithm.")
print("Utilize advanced Q-learning for optimized response generation.")
print("Enter your query, type 'help' for assistance, or 'exit' to end the session.")

def display_help():
  print("🔍 Help - Available Commands:")
  print("  'query [your question]': 🐍 Ask a Python development-related question.")
  print("  'feedback [your feedback]': 🧠 Provide feedback using Q-learning to improve responses.")
  print("  'examples': 📝 Show Python code examples.")
  print("  'debug [your code]': 🐞 Debug your Python code snippet.")
  print("  'exit': 🚪 Exit the session.")
  print("  'help': 🆘 Display this help message.")

# Instantiate a Q-learning agent
q_agent = QLearningAgent(states=30, actions=4)

# Initialize loading_thread to None outside of the try-except block
loading_thread = None

chat_messages = groupchat.messages

# Helper Functions
def process_input(user_input):
    """Process the user input to determine the current state."""
    # Example logic: Using keywords to determine the state
    if "create" in user_input or "python" in user_input:
        return 0  # State for Python-related tasks
    else:
        return 1  # General state for other queries

def quantify_feedback(critic_feedback):
    """Quantify the critic feedback into a numerical reward."""
    # Example logic: Simple sentiment analysis
    positive_feedback_keywords = ['good', 'great', 'excellent']
    if any(keyword in critic_feedback.lower() for keyword in positive_feedback_keywords):
        return 1  # Positive feedback
    else:
        return -1  # Negative or neutral feedback

def determine_next_state(current_state, user_input):
    """Determine the next state based on current state and user input."""
    # Example logic: Alternating states for simplicity
    return (current_state + 1) % q_agent.states

# Main interaction loop
while True:
    try:
        user_input = input("User: ").lower()
        if user_input == "exit":
            break
        elif user_input == "help":
            display_help()
            continue

        # Enhanced state mapping
        current_state = process_input(user_input)

        # Dynamic action choice
        exploration_rate = 0.5
        chosen_action = q_agent.choose_action(current_state, exploration_rate)

        # Execute the chosen action
        loading_thread = start_loading_animation()
        if chosen_action == 0:
            user_proxy.initiate_chat(manager, message=user_input)
        elif chosen_action == 1:
            # Additional logic for assistance based on user_input
            print(f"Providing assistance for: {user_input}")
        elif chosen_action == 2:
            # Additional or alternative actions
            print(f"Performing a specialized task for: {user_input}")
        for message in groupchat.messages[-3:]:
            print(f"{message['sender']}: {message['content']}")
        stop_loading_animation(loading_thread)

        # Critic feedback and Q-learning update
        critic_feedback = input("Critic Feedback (or press Enter to skip): ")
        if critic_feedback:
            reward = quantify_feedback(critic_feedback)
            next_state = determine_next_state(current_state, user_input)
            q_agent.learn(current_state, chosen_action, reward, next_state)

    except Exception as e:
        if loading_thread:
            stop_loading_animation(loading_thread)
        logging.error(str(e))
        print(f"Error: {e}")
