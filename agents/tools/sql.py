import sqlite3
from langchain.tools import Tool
from pydantic import BaseModel
from typing import List

# Connect to a SQLite database named 'db.sqlite'
conn = sqlite3.connect("db.sqlite")

# Function to list all tables in the SQLite database
def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)

# Function to execute a given SQLite query and return the results
def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        # Return an error message if the query fails
        return f"The following error occurred: {str(err)}"

# Pydantic schema for arguments to the `run_sqlite_query` function
class RunQueryArgsSchema(BaseModel):
    query: str

# Create a tool to run SQLite queries using the LangChain framework
run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema
)

# Function to describe the schema of specified tables in the SQLite database
def describe_tables(table_names):
    c = conn.cursor()
    # Format table names for inclusion in the query
    tables = ', '.join("'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return '\n'.join(row[0] for row in rows if row[0] is not None)

# Pydantic schema for arguments to the `describe_tables` function
class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]

# Create a tool to describe table schemas using the LangChain framework
describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema
)
