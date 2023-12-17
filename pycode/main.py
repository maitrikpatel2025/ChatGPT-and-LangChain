# Import necessary modules from langchain, dotenv, and argparse
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse 

# Load environment variables from a .env file
load_dotenv()

# Set up argument parsing for command line inputs
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")  # Default task for the chain
parser.add_argument("--language", default="python")  # Default programming language
args = parser.parse_args()

# Initialize the OpenAI language model
llm = OpenAI()

# Define a code prompt template for generating code based on a task and language
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

# Define a test prompt template for generating test cases for the generated code
test_prompt = PromptTemplate(
    template="Write a test for the following {language} code: {code}",
    input_variables=["language", "code"]
)

# Create a chain for generating code using the language model and code prompt
code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

# Create a chain for generating test cases using the language model and test prompt
test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

# Combine the code and test chains into a sequential chain, processing inputs and outputs in order
chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["test", "code"]
)

# Execute the chain with the provided language and task, and store the result
result = chain({
    "language": args.language,
    "task": args.task
})

# Print the generated code and its corresponding test
print(">>>>>>>>>>>> GENERATED CODE:")
print(result['code'])

print(">>>>>>>>>>>> GENERATED TEST:")
print(result['test'])
