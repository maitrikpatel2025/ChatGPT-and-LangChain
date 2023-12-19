from langchain.tools import StructuredTool
from pydantic import BaseModel

# Define a function to write an HTML report to a file
def write_report(filename, html):
    # Open the specified file in write mode
    with open(filename, 'w') as f:
        f.write(html)  # Write the HTML content to the file

# Create a Pydantic schema for the arguments to the `write_report` function
class WriteReportArgsSchema(BaseModel):
    filename: str  # Name of the file where the report will be saved
    html: str      # HTML content to be written to the file

# Create a StructuredTool using LangChain's framework
# This tool is designed to write HTML reports to a file
write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write an HTML File to disk. Use this tool whenever someone asks for a report.",
    func=write_report,
    args_schema=WriteReportArgsSchema  # Associate the defined argument schema
)
