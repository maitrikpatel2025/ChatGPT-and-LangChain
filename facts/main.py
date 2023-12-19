from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from dotenv import load_dotenv

# Load environment variables from a .env file
# This is typically used for managing configurations and secrets
load_dotenv()

# Initialize embeddings using OpenAI's model
# Embeddings are used to convert text into a numerical vector
embeddings = OpenAIEmbeddings()

# Initialize a text splitter
# This will be used to split large texts into manageable chunks
text_splitter = CharacterTextSplitter(
    separator="\n",       # The character used to split the text
    chunk_size=200,       # The size of each chunk in characters
    chunk_overlap=0       # No overlap between chunks
)

# Load a text file using TextLoader
# The file 'facts.txt' is expected to contain some text data
loader = TextLoader("facts.txt")

# Split the text into chunks using the defined text splitter
# This is useful for processing large documents in smaller parts
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# Create a Chroma database from the split documents
# Chroma uses embeddings to enable efficient text search and similarity comparisons
db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="emb"  # Directory to persist data for future use
)

# Perform a similarity search in the Chroma database
# Searching for documents related to the English language
results = db.similarity_search(
    "What is an interesting fact about the English language?",
)

# Print the content of each search result
for result in results:
    print("\n")
    print(result.page_content)  # Each result is a chunk of text from the database
