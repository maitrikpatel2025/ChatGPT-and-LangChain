from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv

# Import custom tools and handlers
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

# Load environment variables, usually for configuration settings
load_dotenv()

# Initialize a custom handler for chat model events
handler = ChatModelStartHandler()

# Create a ChatOpenAI instance with the defined handler
chat = ChatOpenAI(
    callbacks=[handler]
)

# Retrieve a list of tables from the SQLite database
tables = list_tables()

# Define a prompt template for the chat interaction
# This template sets the context for the AI's capabilities and the database structure
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=(
                "You are an AI that has access to a SQLite database. \n"
                f"The database has tables of: {tables}\n"
                "Do not make any assumptions about what tables exist "
                "or what columns exist. Instead, use the 'describe_tables' function"
                )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

# Initialize a memory buffer for the conversation
# This helps in keeping track of the conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define a list of tools that the agent can use
tools = [
    run_query_tool, 
    describe_tables_tool, 
    write_report_tool
]

# Create an agent capable of executing functions and responding to queries
agent = OpenAIFunctionsAgent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

# Set up an executor for the agent
# This will manage the execution of the agent and its tools
agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,  # Uncomment for detailed logging
    tools=tools,
    memory=memory
)

# Execute the agent with a sample task
agent_executor(
    "Summarize the top 5 most popular products. Write the results to a report file."
)
