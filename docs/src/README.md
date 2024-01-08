![Alt text for the image](https://github.com/ruvnet/q-star/blob/main/DALL%C2%B7E%202023-11-24%2011.21.12%20-%20Create%20an%20artistic%20and%20visually%20appealing%20image%20of%20an%20intelligent%20AI%20agent%20in%20a%20dynamic%20setting,%20without%20any%20text.%20The%20artwork%20should%20convey%20the%20essen.png?raw=true)


# Q-Star Agent: Reinforcement Learning with Microsoft AutoGen

## Introduction
This guide encapsulates my journey in creating intelligent agents, focusing on a reinforcement learning approach, particularly using the Q-Star method. It offers a practical walkthrough of Microsoft's AutoGen library for building and modifying agents. 

The aim is to provide clear instructions from setting up the environment, defining learning capabilities, to managing interactions and inputs. Detailed explanations of each code section are included, making the process transparent and accessible for anyone interested in intelligent agent development.

## Understanding Intelligent Agents

### What Are Intelligent Agents?
Intelligent agents are software entities capable of perceiving their environment autonomously to achieve specific goals. Utilizing advanced large language models like GPT-4, these agents are developed using AutoGen to simplify their creation and enhance capabilities.

### Purpose of Intelligent Agents
Beyond basic automation, these agents in AutoGen aim to orchestrate, optimize, and automate workflows involving LLMs. They integrate with human inputs and tools for complex decision-making, marked by enhanced interaction and conversational intelligence.

## Microsoft AutoGen Overview

### Core Concept
AutoGen leverages advanced LLMs like GPT-4 for creating agents capable of understanding and generating human-like text. It focuses on simplifying the orchestration and automation of LLM workflows.

### Key Features
- **Customizable and Conversable Agents:** AutoGen facilitates the creation of nuanced conversational agents.
- **Integration with Human Inputs:** Enables collaborative solutions combining human expertise and AI efficiency.
- **Multi-Agent Conversations:** Supports scenarios where multiple AI agents interact and collaborate.

## Q-Star and Reinforcement Learning

Q-Star, a variant of Q-learning, is crucial for autonomous decision-making in dynamic environments. It empowers agents to learn and adapt, optimizing their behavior based on experience.

## Introduction to the Q-Star Agent Code Base

### Purpose and Usage
Designed to create and operate intelligent agents using Microsoft's AutoGen library, this code base applies reinforcement learning through the Q-Star approach, suitable for both educational and practical AI projects.

### Key Techniques
- **Reinforcement Learning (Q-Star):** Employs Q-learning for learning optimal actions.
- **Multi-Agent Interaction:** Leverages AutoGen's capability for handling complex agent interactions.
- **User Feedback Integration:** Integrates user inputs and feedback for continuous agent improvement.

## Running the Agent

To configure and run the script with `OAI_CONFIG_LIST.json`, ensuring all dependencies in Docker and Replit, follow these steps:

### Configuring `OAI_CONFIG_LIST.json`

**JSON Configuration:** 
Configure the AutoGen library using the `OAI_CONFIG_LIST.json` file. An example configuration is as follows:

```json
[
    {
        "model": "gpt-4-0314",
        "api_key": "sk-your-key"
    }
]
```
- **`model`**: Set this to the specific model you intend to use, like `"gpt-4-0314"` for a GPT-4 model.
- **`api_key`**: Replace `"sk-your-key"` with your actual OpenAI API key.

**Location of JSON File:**
Place `OAI_CONFIG_LIST.json` in the root directory of your project, where your main Python script is located. Alternatively, adjust the script's path to point to the file's location.

## Running the Script in Docker

### Docker Setup:
1. Ensure Docker is installed on your system. If not, download and install it from the [official Docker website](https://www.docker.com/products/docker-desktop).

### Create a Dockerfile:
2. Write a Dockerfile to define your script's environment. This includes setting the Python version, installing necessary libraries, and copying your script and JSON file into the Docker image.
```Dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install autogen numpy
CMD ["python", "./your_script.py"]
```
1. Building and Running the Docker Image:

Build the Docker image: docker build -t autogen-agent .Run the Docker container: docker run -it --rm autogen-agent

## Running the Script in Replit

- Replit Setup:

Create a new Python project in Replit.Upload your script and the OAI_CONFIG_LIST.json file to the project.

- Dependencies:

Add necessary dependencies (like autogen, numpy) to a requirements.txt file, or install them directly in the Replit shell.

Example requirements.txt:
```txt
autogen
numpy
```
### Environment Variables:
- Check for the `REPL_ID` environment variable in your script to identify if it's running in Replit. Set this variable in the Replit environment as needed.

### Running the Script:
- Run the script directly in the Replit interface.

By following these steps, you can configure and run your script with the AutoGen library in both Docker and Replit environments, using settings from the `OAI_CONFIG_LIST.json` file. Remember to handle API keys securely, avoiding exposure in public repositories.

## Code Structure

The code is organized into distinct sections, each with a specific role:

### Library Imports and Environment Setup:
- Importing necessary libraries.
- Setting up the environment, crucial for the rest of the code.

### Q-Learning Agent Definition:
- Define the Q-learning agent, key to the reinforcement learning process.
- The agent learns from its environment to make decisions, using the Q-Star algorithm.

### ASCII Loading Animation:
- Implement a loading animation to visually indicate processing or waiting times, enhancing user interaction.

### AutoGen Configuration and Agent Creation:
- Set up the AutoGen framework.
- Configure agents, initialize the group chat, and manage agent interactions.

### User Interaction Loop:
- Handle real-time user inputs in the main loop.
- Process inputs and update the agent's learning based on feedback.

### Error Handling:
- Robust error handling includes catching and logging exceptions to ensure code stability.

  ### Step 1. Importing Libraries

```python
import os
import autogen
from autogen import config_list_from_json, UserProxyAgent, AssistantAgent, GroupChatManager, GroupChat
import numpy as np
import random
import logging
import threading
import sys
import time
```
- **`os`**: Provides functions for interacting with the operating system.
- **`autogen`**: The core library for creating intelligent agents.
- **`config_list_from_json`, `UserProxyAgent`, `AssistantAgent`, `GroupChatManager`, `GroupChat`**: Specific components from the Autogen library used in the agent's setup.
- **`numpy`** (np): Supports large, multi-dimensional arrays and matrices, along with a vast collection of high-level mathematical functions.
- **`random`**: Implements pseudo-random number generators for various distributions.
- **`logging`**: Facilitates logging events into a file or other outputs.
- **`threading`**: Allows the creation of thread-based parallelism.
- **`sys`, `time`**: Provides access to some variables used by the interpreter (`sys`) and time-related functions (`time`).

## Step 2: Setting Up the Script and Logging
```python
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
```
- Determines the directory of the script for relative file paths.
- Sets up a log file to capture errors.
- Checks the environment (Replit or non-Replit) and adjusts settings accordingly.

## Step 3: Defining the Q-Learning Agent
```python
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
```
- Initialization (__init__): Sets up states, actions, learning parameters, and initializes the Q-table.
- choose_action: Decides whether to explore (choose randomly) or exploit (use the best known action).
- learn: Updates the Q-table based on the agent's experiences.

## Step 4: ASCII Loading Animation

```python
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
```
- frames: Defines the visual frames for the loading animation.
- animate_loading: Handles the continuous display and update of the loading frames.
- start_loading_animation and stop_loading_animation: Start and stop the animation in a separate thread.

## AutoGen Configuration and Agent Setup
```python
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
```
- Loads the AutoGen configuration from a JSON file.
- Initializes user and assistant agents with specific configurations.
- Creates a group chat and a group chat manager to facilitate interactions.

## User Interaction and Main Loop
```python
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
```
- This snippet displays the ASCII art representing "Q*", symbolizing the Q-Star algorithm.
- It introduces the user to the Q-Star Agent, highlighting its use of advanced Q-learning.
- Instructions are provided for interaction, such as entering queries, seeking help, or exiting the session.

## display_help Function
```python
def display_help():
  print("🔍 Help - Available Commands:")
  print("  'query [your question]': 🐍 Ask a Python development-related question.")
  print("  'feedback [your feedback]': 🧠 Provide feedback using Q-learning to improve responses.")
  print("  'examples': 📝 Show Python code examples.")
  print("  'debug [your code]': 🐞 Debug your Python code snippet.")
  print("  'exit': 🚪 Exit the session.")
  print("  'help': 🆘 Display this help message.")
```

- This function lists available commands for the user.
- Commands include asking questions, providing feedback, viewing examples, debugging code, exiting the session, and displaying the help message again.

##Instantiating the Q-Learning Agent
```python
# Instantiate a Q-learning agent
q_agent = QLearningAgent(states=30, actions=4)
```
- Creates an instance of the Q-learning agent with specified states and actions.
- This agent is essential for the reinforcement learning part of the program.

## Initialization of loading_thread and chat_messages
```python
# Initialize loading_thread to None outside of the try-except block
loading_thread = None

chat_messages = groupchat.messages
```
- Initializes loading_thread to None. This variable will later control the ASCII loading animation.
- chat_messages holds the messages from the group chat, facilitating communication between agents.

## Helper Functions
```python
def process_input(user_input):
    """Process the user input to determine the current state."""
    if "create" in user_input or "python" in user_input:
        return 0  # State for Python-related tasks
    else:
        return 1  # General state for other queries

def quantify_feedback(critic_feedback):
    """Quantify the critic feedback into a numerical reward."""
    positive_feedback_keywords = ['good', 'great', 'excellent']
    if any(keyword in critic_feedback.lower() for keyword in positive_feedback_keywords):
        return 1  # Positive feedback
    else:
        return -1  # Negative or neutral feedback

def determine_next_state(current_state, user_input):
    """Determine the next state based on current state and user input."""
    return (current_state + 1) % q_agent.states
```
- process_input: Analyzes user input to determine the current state of the agent.
- quantify_feedback: Converts critic feedback into numerical rewards for the Q-learning algorithm.
- determine_next_state: Calculates the next state based on the current state and user input, crucial for the agent's learning process.

## Main Interaction Loop
```python
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
```
- This loop is the core of user interaction, handling inputs and directing the flow of the program.
- Handles user commands and uses the Q-learning agent to determine actions.
- Manages the loading animation and processes feedback to update the Q-learning agent.
- The loop continues indefinitely until the user decides to exit.

## Exception handling block

```python
except Exception as e:
    if loading_thread:
        stop_loading_animation(loading_thread)
    logging.error(str(e))
    print(f"Error: {e}")
```
```python
except Exception as e:
    # This line catches any kind of exception that occurs in the preceding try block.
    # 'Exception' is a base class for all built-in exceptions in Python, excluding
    # system exit exceptions and keyboard interruptions. 'as e' assigns the 
    # exception object to the variable 'e', which can be used to get more information 
    # about the error.

    if loading_thread:
        # Checks if the 'loading_thread' variable is not None. If it exists, it 
        # implies that the loading animation is currently active.

        stop_loading_animation(loading_thread)
        # Calls 'stop_loading_animation' with 'loading_thread' as an argument.
        # This function stops the loading animation safely, ensuring proper termination 
        # of the thread handling the animation.

    logging.error(str(e))
    # Logs the error message to a file or another logging destination.
    # 'str(e)' converts the exception object to a string describing the error,
    # important for debugging and understanding the underlying issues.

    print(f"Error: {e}")
    # Prints the error message to the standard output (console).
    # Uses an f-string format where '{e}' is replaced by the error's string representation.
    # Provides immediate feedback to the user about the error.
```
