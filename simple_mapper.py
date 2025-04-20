import os
import json
import re
import logging
import openai
from dotenv import load_dotenv
import time
import uuid
import traceback
from collections import deque

# For Google Sheets integration
try:
    from sheets_manager import extract_spreadsheet_id, get_sheet_structure, create_new_sheet, update_sheet_data, get_sheet_data
    SHEETS_MANAGER_AVAILABLE = True
except ImportError:
    SHEETS_MANAGER_AVAILABLE = False
    logging.warning("sheets_manager module not available. Google Sheets functionality will be limited.")

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class QuerySession:
    """Class to manage an ongoing query session with context memory"""
    
    def __init__(self, financial_data=None, matrix=None, filled_matrix=None, sheet_url=None):
        self.session_id = str(uuid.uuid4())
        self.financial_data = financial_data
        self.matrix = matrix
        self.filled_matrix = filled_matrix
        self.sheet_url = sheet_url
        self.message_history = deque(maxlen=20)  # Store the last 20 messages
        self.context = self._build_initial_context()
        
    def _build_initial_context(self):
        """Build the initial context based on available data"""
        context_parts = []
        
        if self.financial_data:
            context_parts.append("FINANCIAL DATA AVAILABLE: Yes")
            if self.financial_data.get('metadata'):
                doc_type = self.financial_data.get('metadata', {}).get('document_type', 'Unknown')
                context_parts.append(f"DOCUMENT TYPE: {doc_type}")
            
            num_tables = len(self.financial_data.get('tables', []))
            num_kv_pairs = len(self.financial_data.get('key_value_pairs', []))
            context_parts.append(f"DATA SUMMARY: {num_tables} tables, {num_kv_pairs} key-value pairs")
        else:
            context_parts.append("FINANCIAL DATA AVAILABLE: No")
        
        if self.matrix:
            matrix_rows = len(self.matrix)
            matrix_cols = len(self.matrix[0]) if matrix_rows > 0 else 0
            context_parts.append(f"TEMPLATE MATRIX: {matrix_rows}x{matrix_cols} matrix available")
        else:
            context_parts.append("TEMPLATE MATRIX: Not available")
        
        if self.filled_matrix:
            filled_rows = len(self.filled_matrix)
            filled_cols = len(self.filled_matrix[0]) if filled_rows > 0 else 0
            context_parts.append(f"FILLED MATRIX: {filled_rows}x{filled_cols} matrix available")
        else:
            context_parts.append("FILLED MATRIX: Not available")
        
        if self.sheet_url:
            context_parts.append(f"GOOGLE SHEET: Connected to {self.sheet_url}")
        else:
            context_parts.append("GOOGLE SHEET: Not connected")
            
        return "\n".join(context_parts)
    
    def add_message(self, role, content):
        """Add a message to the conversation history"""
        self.message_history.append({"role": role, "content": content})
    
    def get_messages_for_api(self):
        """Get the messages in a format suitable for the OpenAI API"""
        # First message is always the system message with context
        system_message = {
            "role": "system", 
            "content": self._get_system_prompt()
        }
        
        # Convert the deque to a list
        history = list(self.message_history)
        
        # Return the combined messages
        return [system_message] + history
    
    def _get_system_prompt(self):
        """Get the system prompt with full context"""
        return f"""You are a financial data agent that helps users interact with their financial data and Google Sheets.
You can answer questions about the data, suggest actions, and help execute operations on Google Sheets.

CURRENT SESSION CONTEXT:
{self.context}

You have these capabilities:
1. Answer questions about the financial data
2. Explain the structure of the matrix/template
3. Help migrate data to Google Sheets
4. Make specific changes to cells, rows, or columns
5. Execute operations like adding, updating, or calculating values

When the user asks you to perform an action on Google Sheets, respond with:
1. A clear explanation of what you'll do
2. The specific action details in a format that can be executed
3. Any warnings or considerations they should know

Be conversational but precise. Reference specific cells by their coordinates (e.g., A1, B5).
If asked about data not available in the context, politely explain the limitation.

If asked to execute a Google Sheets operation, format your response like this:
ACTION: [add_value/update_cell/migrate_data/etc.]
PARAMETERS: {{"sheet_name": "Sheet1", "cell": "B5", "value": "123"}}
EXPLANATION: Brief explanation of what this will do

This allows the system to parse your response and execute the requested operation."""

def process_query(session, query):
    """
    Process a user query in the context of the ongoing session
    
    Args:
        session (QuerySession): Current query session with context
        query (str): User's query/question
        
    Returns:
        dict: Response with answer and potentially action to perform
    """
    try:
        # Add the user's message to the history
        session.add_message("user", query)
        
        # Call OpenAI with the full conversation history
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
        
        openai.api_key = api_key
        
        # Get all messages for the API call
        messages = session.get_messages_for_api()
        
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using a current model with context memory
            messages=messages,
            temperature=0.7,  # More creative for conversation
            max_tokens=8000
        )
        
        # Extract the response content
        content = response.choices[0].message['content'].strip()
        
        # Add the assistant's response to the history
        session.add_message("assistant", content)
        
        # Parse the response for any actions
        action_info = parse_action_from_response(content)
        
        return {
            "response": content,
            "action": action_info
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "response": f"I'm sorry, there was an error processing your query: {str(e)}",
            "action": None
        }

def parse_action_from_response(response):
    """
    Parse the response to extract action information
    
    Args:
        response (str): Response from the agent
        
    Returns:
        dict or None: Extracted action information or None if no action found
    """
    # Look for action blocks in the format:
    # ACTION: action_name
    # PARAMETERS: {json_parameters}
    # EXPLANATION: explanation text
    
    action_match = re.search(r'ACTION:\s*(\w+)', response)
    if not action_match:
        return None
    
    action_name = action_match.group(1)
    
    # Extract parameters (as JSON)
    params_match = re.search(r'PARAMETERS:\s*({.*?})', response, re.DOTALL)
    parameters = {}
    if params_match:
        try:
            parameters = json.loads(params_match.group(1))
        except json.JSONDecodeError:
            # If it's not valid JSON, try to extract key-value pairs
            param_text = params_match.group(1)
            param_matches = re.findall(r'"([^"]+)":\s*"([^"]+)"', param_text)
            for key, value in param_matches:
                parameters[key] = value
    
    # Extract explanation
    explanation_match = re.search(r'EXPLANATION:(.*?)(?:ACTION:|$)', response, re.DOTALL)
    explanation = explanation_match.group(1).strip() if explanation_match else ""
    
    return {
        "action": action_name,
        "parameters": parameters,
        "explanation": explanation
    }

def execute_action(session, action_info):
    """
    Execute the requested action
    
    Args:
        session (QuerySession): Current query session
        action_info (dict): Action details extracted from response
        
    Returns:
        dict: Result of the action execution
    """
    if not action_info:
        return {"success": False, "message": "No action to execute"}
    
    action = action_info.get("action", "").lower()
    params = action_info.get("parameters", {})
    
    try:
        if action == "migrate_data":
            # Migrate data to Google Sheets
            return migrate_data_to_sheet(session, params)
        
        elif action == "update_cell":
            # Update a specific cell
            sheet_name = params.get("sheet_name")
            cell = params.get("cell")
            value = params.get("value")
            
            if not all([sheet_name, cell, value is not None]):
                return {"success": False, "message": "Missing required parameters for update_cell"}
            
            return update_cell_in_sheet(session, sheet_name, cell, value)
        
        elif action == "add_value":
            # Add a value to a specific location
            sheet_name = params.get("sheet_name")
            row = params.get("row")
            col = params.get("col")
            value = params.get("value")
            
            if not all([sheet_name, row is not None, col is not None, value is not None]):
                return {"success": False, "message": "Missing required parameters for add_value"}
            
            return add_value_to_sheet(session, sheet_name, row, col, value)
        
        elif action == "connect_sheet":
            # Connect to a Google Sheet
            sheet_url = params.get("sheet_url")
            
            if not sheet_url:
                return {"success": False, "message": "Missing sheet_url parameter"}
            
            return connect_to_sheet(session, sheet_url)
        
        else:
            return {"success": False, "message": f"Unknown action: {action}"}
    
    except Exception as e:
        logger.error(f"Error executing action: {str(e)}")
        logger.error(traceback.format_exc())
        return {"success": False, "message": f"Error executing action: {str(e)}"}

def migrate_data_to_sheet(session, params):
    """
    Migrate data to a Google Sheet
    
    Args:
        session (QuerySession): Current query session
        params (dict): Parameters for the migration
        
    Returns:
        dict: Result of the migration
    """
    if not SHEETS_MANAGER_AVAILABLE:
        return {"success": False, "message": "Google Sheets integration is not available"}
    
    if not session.sheet_url:
        return {"success": False, "message": "No Google Sheet connected. Please connect to a sheet first."}
    
    if not session.filled_matrix:
        return {"success": False, "message": "No filled matrix available to migrate"}
    
    sheet_name = params.get("sheet_name", "MigratedData")
    create_new = params.get("create_new", True)
    
    try:
        spreadsheet_id = extract_spreadsheet_id(session.sheet_url)
        
        if create_new:
            # Create a new sheet
            from googleapiclient.discovery import build
            from sheets_manager import get_credentials
            
            creds = get_credentials()
            service = build('sheets', 'v4', credentials=creds)
            
            result = create_new_sheet(
                service,
                spreadsheet_id,
                sheet_name,
                headers=None,  # No separate headers, everything is in the matrix
                data=session.filled_matrix
            )
            
            return {
                "success": True, 
                "message": f"Successfully created new sheet '{sheet_name}' with the filled matrix data",
                "sheet_name": sheet_name
            }
        else:
            # Update existing sheet
            success = update_sheet_data(spreadsheet_id, sheet_name, session.filled_matrix)
            
            if success:
                return {
                    "success": True,
                    "message": f"Successfully updated sheet '{sheet_name}' with the filled matrix data",
                    "sheet_name": sheet_name
                }
            else:
                return {"success": False, "message": f"Failed to update sheet '{sheet_name}'"}
    
    except Exception as e:
        logger.error(f"Error migrating data to sheet: {str(e)}")
        return {"success": False, "message": f"Error migrating data to sheet: {str(e)}"}

def update_cell_in_sheet(session, sheet_name, cell, value):
    """
    Update a specific cell in the Google Sheet
    
    Args:
        session (QuerySession): Current query session
        sheet_name (str): Name of the sheet
        cell (str): Cell reference (e.g., "A1")
        value: Value to set
        
    Returns:
        dict: Result of the update
    """
    if not SHEETS_MANAGER_AVAILABLE:
        return {"success": False, "message": "Google Sheets integration is not available"}
    
    if not session.sheet_url:
        return {"success": False, "message": "No Google Sheet connected. Please connect to a sheet first."}
    
    try:
        spreadsheet_id = extract_spreadsheet_id(session.sheet_url)
        
        # Get the current sheet data
        sheet_data = get_sheet_data(spreadsheet_id, sheet_name)
        
        if not sheet_data:
            return {"success": False, "message": f"Sheet '{sheet_name}' not found or is empty"}
        
        # Parse the cell reference to get row and column
        col_letter = ''.join(c for c in cell if c.isalpha()).upper()
        row_num = int(''.join(c for c in cell if c.isdigit())) - 1  # Convert to 0-based
        
        # Convert column letter to number (A=0, B=1, etc.)
        col_num = 0
        for i, char in enumerate(reversed(col_letter)):
            col_num += (ord(char) - ord('A') + 1) * (26 ** i)
        col_num -= 1  # Convert to 0-based
        
        # Ensure the matrix is large enough
        while len(sheet_data) <= row_num:
            sheet_data.append([])
        
        for i in range(len(sheet_data)):
            while len(sheet_data[i]) <= col_num:
                sheet_data[i].append("")
        
        # Update the cell
        sheet_data[row_num][col_num] = value
        
        # Update the sheet
        success = update_sheet_data(spreadsheet_id, sheet_name, sheet_data)
        
        if success:
            # Update the filled matrix in the session if it matches the updated sheet
            if session.filled_matrix and sheet_name == params.get("sheet_name", "Sheet1"):
                while len(session.filled_matrix) <= row_num:
                    session.filled_matrix.append([])
                for i in range(len(session.filled_matrix)):
                    while len(session.filled_matrix[i]) <= col_num:
                        session.filled_matrix[i].append("")
                session.filled_matrix[row_num][col_num] = value
            
            return {
                "success": True,
                "message": f"Successfully updated cell {cell} in sheet '{sheet_name}' to {value}"
            }
        else:
            return {"success": False, "message": f"Failed to update cell {cell} in sheet '{sheet_name}'"}
    
    except Exception as e:
        logger.error(f"Error updating cell: {str(e)}")
        return {"success": False, "message": f"Error updating cell: {str(e)}"}

def add_value_to_sheet(session, sheet_name, row, col, value):
    """
    Add a value to a specific location in the Google Sheet
    
    Args:
        session (QuerySession): Current query session
        sheet_name (str): Name of the sheet
        row (int): Row index (0-based)
        col (int): Column index (0-based)
        value: Value to add
        
    Returns:
        dict: Result of the addition
    """
    if not SHEETS_MANAGER_AVAILABLE:
        return {"success": False, "message": "Google Sheets integration is not available"}
    
    if not session.sheet_url:
        return {"success": False, "message": "No Google Sheet connected. Please connect to a sheet first."}
    
    try:
        # Convert to 0-based indices if they are 1-based
        row_idx = int(row) - 1 if isinstance(row, (int, str)) and str(row).isdigit() else row
        
        # If col is a letter (A, B, C), convert it to a number
        if isinstance(col, str) and col.isalpha():
            col_idx = 0
            for i, char in enumerate(reversed(col.upper())):
                col_idx += (ord(char) - ord('A') + 1) * (26 ** i)
            col_idx -= 1  # Convert to 0-based
        else:
            col_idx = int(col) - 1 if isinstance(col, (int, str)) and str(col).isdigit() else col
        
        # Get the spreadsheet ID
        spreadsheet_id = extract_spreadsheet_id(session.sheet_url)
        
        # Get the current sheet data
        sheet_data = get_sheet_data(spreadsheet_id, sheet_name)
        
        if not sheet_data:
            # Sheet doesn't exist, create a new one
            from googleapiclient.discovery import build
            from sheets_manager import get_credentials
            
            creds = get_credentials()
            service = build('sheets', 'v4', credentials=creds)
            
            # Create a new empty sheet
            create_new_sheet(service, spreadsheet_id, sheet_name)
            
            # Initialize with empty data
            sheet_data = [[]]
        
        # Ensure the matrix is large enough
        while len(sheet_data) <= row_idx:
            sheet_data.append([])
        
        for i in range(len(sheet_data)):
            while len(sheet_data[i]) <= col_idx:
                sheet_data[i].append("")
        
        # Add the value
        sheet_data[row_idx][col_idx] = value
        
        # Update the sheet
        success = update_sheet_data(spreadsheet_id, sheet_name, sheet_data)
        
        if success:
            # Update the filled matrix in the session if it matches the updated sheet
            if session.filled_matrix and sheet_name == "Sheet1":  # Assuming Sheet1 is the main sheet
                while len(session.filled_matrix) <= row_idx:
                    session.filled_matrix.append([])
                for i in range(len(session.filled_matrix)):
                    while len(session.filled_matrix[i]) <= col_idx:
                        session.filled_matrix[i].append("")
                session.filled_matrix[row_idx][col_idx] = value
            
            # Convert indices to user-friendly format
            col_letter = ""
            temp_col = col_idx + 1
            while temp_col > 0:
                temp_col, remainder = divmod(temp_col - 1, 26)
                col_letter = chr(65 + remainder) + col_letter
            
            cell_ref = f"{col_letter}{row_idx + 1}"
            
            return {
                "success": True,
                "message": f"Successfully added value {value} to cell {cell_ref} in sheet '{sheet_name}'"
            }
        else:
            return {"success": False, "message": f"Failed to add value to sheet '{sheet_name}'"}
    
    except Exception as e:
        logger.error(f"Error adding value to sheet: {str(e)}")
        return {"success": False, "message": f"Error adding value to sheet: {str(e)}"}

def connect_to_sheet(session, sheet_url):
    """
    Connect to a Google Sheet
    
    Args:
        session (QuerySession): Current query session
        sheet_url (str): URL of the Google Sheet
        
    Returns:
        dict: Result of the connection
    """
    if not SHEETS_MANAGER_AVAILABLE:
        return {"success": False, "message": "Google Sheets integration is not available"}
    
    try:
        # Try to get the structure of the sheet to verify it exists and is accessible
        structure = get_sheet_structure(sheet_url)
        
        if not structure:
            return {"success": False, "message": "Could not connect to the Google Sheet. Please check the URL and permissions."}
        
        # Update the session with the sheet URL
        session.sheet_url = sheet_url
        session.context = session._build_initial_context()
        
        return {
            "success": True,
            "message": f"Successfully connected to Google Sheet: {structure.get('title', sheet_url)}",
            "sheet_structure": structure
        }
    
    except Exception as e:
        logger.error(f"Error connecting to sheet: {str(e)}")
        return {"success": False, "message": f"Error connecting to sheet: {str(e)}"}

def interactive_query_processor(financial_data=None, matrix=None, filled_matrix=None, sheet_url=None):
    """
    Start an interactive query processing session
    
    Args:
        financial_data (dict, optional): Financial data extracted from PDF
        matrix (list, optional): Original matrix template
        filled_matrix (list, optional): Matrix filled with financial data
        sheet_url (str, optional): URL of the Google Sheet to connect to
        
    Returns:
        None
    """
    # Initialize the session
    session = QuerySession(financial_data, matrix, filled_matrix, sheet_url)
    
    print("\n=== Financial Data Query Agent ===")
    print("Ask questions about your financial data, request operations on Google Sheets,")
    print("or ask for help with migration. Type 'exit' to end the session.\n")
    
    # Print initial session info
    print(session.context)
    print("\nSession ID:", session.session_id)
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            query = input("\nYou: ")
            
            # Check for exit command
            if query.lower() in ['exit', 'quit', 'bye']:
                print("\nEnding session. Goodbye!")
                break
            
            # Process the query
            result = process_query(session, query)
            
            # Display the response
            print("\nAgent:", result["response"])
            
            # Execute any action if present
            action_info = result.get("action")
            if action_info:
                print("\nExecuting action:", action_info["action"])
                action_result = execute_action(session, action_info)
                
                if action_result["success"]:
                    print("✅", action_result["message"])
                else:
                    print("❌", action_result["message"])
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Let's continue with a new query.")

def format_pdf_data_as_context(financial_data):
    """
    Format the extracted PDF data as a structured context for the mapping API
    
    Args:
        financial_data (dict): Extracted financial data from PDF
        
    Returns:
        str: Formatted context with tables and key-value pairs
    """
    context = []
    
    # Start with document metadata
    metadata = financial_data.get('metadata', {})
    if metadata:
        context.append("=== DOCUMENT METADATA ===")
        for key, value in metadata.items():
            context.append(f"{key}: {value}")
        context.append("")
    
    # Add tables with HTML content
    for i, table in enumerate(financial_data.get('tables', []), 1):
        table_type = table.get('type', f'Table {i}')
        context.append(f"TABLE {i}: {table_type}")
        context.append("----------------------------------------")
        context.append("")
        
        # Add HTML table if available
        if 'html' in table:
            context.append(table['html'])
        else:
            # Create HTML table if not available
            headers = table.get('headers', [])
            data = table.get('data', [])
            
            html = "<table>\n"
            
            # Add headers
            if headers:
                html += "  <thead>\n    <tr>\n"
                for header in headers:
                    html += f"      <th>{header}</th>\n"
                html += "    </tr>\n  </thead>\n"
            
            # Add data rows
            html += "  <tbody>\n"
            for row in data:
                html += "    <tr>\n"
                for cell in row:
                    html += f"      <td>{cell}</td>\n"
                html += "    </tr>\n"
            html += "  </tbody>\n"
            
            html += "</table>"
            context.append(html)
        
        context.append("")
        context.append("----------------------------------------")
        context.append("")
    
    # Add key-value pairs
    if financial_data.get('key_value_pairs'):
        context.append("=== KEY VALUE PAIRS ===")
        context.append("----------------------------------------")
        context.append("")
        
        for i, kv in enumerate(financial_data.get('key_value_pairs', []), 1):
            key = kv.get('key', '')
            value = kv.get('value', '')
            context.append(f"{i}. {key}: {value}")
        
        context.append("")
        context.append("----------------------------------------")
    
    return "\n".join(context)

def format_matrix_for_prompt(matrix):
    """
    Format a matrix in a human-readable way for the prompt
    
    Args:
        matrix (list): List of lists representing the matrix
        
    Returns:
        str: Human-readable formatted matrix
    """
    result = []
    
    for row in matrix:
        formatted_row = []
        for cell in row:
            if cell == "" or cell is None:
                formatted_row.append('""')
            elif isinstance(cell, str):
                formatted_row.append(f'"{cell}"')
            else:
                formatted_row.append(str(cell))
        result.append("[" + ", ".join(formatted_row) + "]")
    
    return "[\n  " + ",\n  ".join(result) + "\n]"

def normalize_matrix(matrix):
    """
    Normalize a matrix to ensure all rows have the same number of columns
    
    Args:
        matrix (list): List of lists representing the matrix
        
    Returns:
        list: Normalized matrix with all rows having the same number of columns
    """
    if not matrix:
        return []
    
    # Find the maximum number of columns
    max_cols = 0
    for row in matrix:
        max_cols = max(max_cols, len(row) if row else 0)
    
    # Normalize each row to have the same number of columns
    normalized_matrix = []
    for row in matrix:
        if not row:
            normalized_matrix.append([""] * max_cols)
        else:
            normalized_row = row.copy()
            while len(normalized_row) < max_cols:
                normalized_row.append("")
            normalized_matrix.append(normalized_row)
    
    return normalized_matrix

def validate_filled_matrix(filled_matrix, financial_data, context):
    """
    Validate the filled matrix with another OpenAI call to verify accuracy
    
    Args:
        filled_matrix (list): Matrix filled with financial data
        financial_data (dict): Original financial data
        context (str): Formatted context
        
    Returns:
        dict: Validation results with possible corrections
    """
    # Set up OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
    
    openai.api_key = api_key
    
    # Format the filled matrix for the prompt
    matrix_formatted = format_matrix_for_prompt(filled_matrix)
    
    # Create validation prompt
    system_prompt = """You are a financial data validation expert. Your task is to verify that a matrix has been correctly filled with financial data from a context and suggest improvements if needed.

IMPORTANT: You must preserve the EXACT structure of the original matrix - only suggest changes to cell values, never add or remove rows or columns.

IF YOU THINK ANY CELL CAN BE FILLED THE DATA THAT IS MISSING FROM THE MATRIX OR EMPTY THEN ADD IT TO THE OUTPUT MATRIX 
THAT IS VALIDATE AND ALSO FILL THE MATRIX ACCORDINGLY

Examine both the filled matrix and the context information to:
1. Verify that values in the matrix match the data in the context
2. Identify any missing values that could be added from the context
3. Check that header information is correct and complete
4. Ensure that all numerical data is placed in the correct cells
5. Validate that percentages and calculations are accurate

Provide your response as a well-structured JSON object with the following fields:
1. "is_matrix_correct": A clear Yes/No answer
2. "cells_changed": Number of cells you modified (if any)
3. "validation_notes": Brief explanation of your findings and changes
4. "corrected_matrix": The matrix with your corrections (if any)

If the matrix is already correct and optimally filled, simply return that it's correct.

HERE IS THE CONTEXT:
READ ANALYSE AND UNDERSTAND THE CONTEXT THEN VALIDATE THE MATRIX GIVEN BY THE USER

CONTEXT INFORMATION (source of truth):
----------------------------------------

{context}

"""

    user_prompt = f"""VALIDATE THE FOLLOWING MATRIX AGAINST THE FINANCIAL CONTEXT

I have a filled matrix of financial data that was automatically mapped from the context. Please verify its accuracy and completeness.

FILLED MATRIX TO VALIDATE:
{matrix_formatted}

Please examine the matrix against the context and verify:
1. Are all values correctly mapped from the context?
2. Are there any cells that could be filled with data from the context but are currently empty?
3. Are headers properly labeled?
4. Are any numerical values incorrect or placed in the wrong cells?
5. Are percentages and calculations accurate?

Return your response as a valid JSON object with the fields:
- "is_matrix_correct": "Yes" or "No"
- "cells_changed": Number of cells changed (0 if matrix is already correct)
- "validation_notes": Brief explanation of your findings
- "corrected_matrix": Only include if you made changes

DO NOT ALTER THE STRUCTURE of the matrix (don't add or remove rows/columns)."""

    logger.info("Making validation API call to OpenAI")
    
    try:
        # Make the API call for validation
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using a current model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=8000,
            response_format={"type": "json_object"}  # Force JSON response format
        )
        
        # Save the raw API response and processed content
        raw_response = response
        content = response.choices[0].message['content'].strip()
        
        # Save debugging information to a file
        debug_data = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_api_response": str(raw_response),
            "response_content": content
        }
        
        validation_debug_file = f"validation_debug_{int(time.time())}.json"
        try:
            with open(validation_debug_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
            logger.info(f"Validation debug data saved to {validation_debug_file}")
        except Exception as debug_err:
            logger.warning(f"Could not save validation debug data: {debug_err}")
        
        # Parse the validation response
        validation_result = json.loads(content)
        validation_result['debug_file'] = validation_debug_file
        
        return validation_result
    
    except Exception as e:
        logger.error(f"Error validating matrix: {str(e)}")
        return {
            "is_matrix_correct": "Error",
            "cells_changed": 0,
            "validation_notes": f"Error during validation: {str(e)}",
            "debug_file": validation_debug_file if 'validation_debug_file' in locals() else None
        }

def map_financial_data(financial_data, matrix):
    """
    Map financial data to a matrix using OpenAI's GPT model
    
    Args:
        financial_data (dict): Extracted financial data from PDF
        matrix (list): Matrix template to fill with data
        
    Returns:
        dict: Mapping results with filled matrix
    """
    # Set up OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
    
    openai.api_key = api_key
    
    # Format the financial data as a structured context
    context = format_pdf_data_as_context(financial_data)
    logger.info(f"Created context with {len(context)} characters")
    
    # Normalize the matrix
    normalized_matrix = normalize_matrix(matrix)
    matrix_formatted = format_matrix_for_prompt(normalized_matrix)
    
    # Create system prompt with matrix included
    system_prompt = f"""YOU ARE A FINANCIAL DATA EXPERT - YOUR TASK IS TO FILL IN AS MANY CELLS AS POSSIBLE IN A FINANCIAL MATRIX FROM THE GIVEN CONTEXT

I WILL PROVIDE TWO COMPONENTS:

1. A MATRIX TEMPLATE that contains headers and structure for a financial sheet — most cells will be empty or partially filled.
2. A CONTEXT SECTION containing tables and key-value pairs with relevant financial data.

YOUR TASK IS TO:

- FILL IN AS MANY CELLS AS POSSIBLE IN THE MATRIX I WANT THE MATRIX TO BE FULLY FILLED OR MAXIMUM POSSIBLE IN THE GIVEN CONTEXT

1. CAREFULLY READ AND ANALYZE THE CONTEXT
2. UNDERSTAND THE FULL STRUCTURE AND INTENTION BEHIND THE MATRIX
3. FILL IN ALL MISSING VALUES WITH RELEVANT DATA FROM THE CONTEXT EVEN IF SOME CALCULATIONS NEEDED
4. LEAVE NO CELL EMPTY IF THERE IS CORRESPONDING DATA IN THE CONTEXT
5. MAP EVERY FINANCIAL DATA POINT FROM THE CONTEXT TO THE APPROPRIATE CELL IN THE MATRIX
6. VALIDATE THE MATRIX AGAINST THE CONTEXT TO ENSURE ACCURACY AND COMPLETENESS

PAY SPECIFIC ATTENTION TO THE FOLLOWING:


##### ADD ALL NECCESSARY YEARS TO THE MATRIX AS HEADERS IN DATE RELATED ROWS ABOVE THE NUMBER OF MONTHS IF ANY #####


- DONOT INVENT ANY DATA - ONLY USE DATA PROVIDED IN THE CONTEXT
- ONLY FILL IN THE MISSING CELLS ACCURATELY PERFORM ANY CALUCLATIONS ON GIVE DATA IF NEEDED
- RETURN THE MATRIX IN THE EXACT SAME STRUCTURE AS RECEIVED — DO NOT CHANGE FORMATTING, HEADERS, OR LAYOUT. ONLY FILL IN THE MISSING CELLS ACCURATELY.

FINANCIAL PERCENTAGES SHOULD BE CALCULATED RELATIVE TO TOTAL OPERATING INCOME/REVENUE.

YOUR OUTPUT MUST INCLUDE EVERY SINGLE REVENUE, EXPENSE, AND PROFIT ITEM FROM THE INCOME STATEMENT IN THE CONTEXT.

DO NOT HALLUCINATE OR INVENT DATA THAT IS NOT PRESENT IN THE CONTEXT.

RETURN THE MATRIX IN THE EXACT SAME STRUCTURE AS RECEIVED — DO NOT CHANGE FORMATTING, HEADERS, OR LAYOUT. ONLY FILL IN THE MISSING CELLS ACCURATELY.

THE FINAL OUTPUT SHOULD RESEMBLE A CLEAN, PROFESSIONAL FINANCIAL SHEET WITH MAXIMUM DATA FILLED IN FROM THE CONTEXT.

PROVIDE YOUR RESPONSE AS A JSON OBJECT CONTAINING:
1. "Filled Matrix": The matrix with all values filled in (exactly the same structure as provided)
2. "Number of cells filled": Count of cells with financial data added
3. "Number of cells not filled": Count of cells still empty after processing

HERE IS THE MATRIX:
{matrix_formatted}"""

    # Create user prompt without the matrix
    user_prompt = f"""HERE IS SOME CONTEXT WITH TABLES AND KEY-VALUE PAIRS. PLEASE FILL AS MANY CELLS AS POSSIBLE IN THE MATRIX USING THE DATA PROVIDED.

YOUR PRIMARY GOAL IS TO MAXIMIZE THE NUMBER OF CELLS FILLED WITH ACCURATE DATA. LEAVE NO CELL EMPTY IF MATCHING DATA EXISTS.

CONVERT ALL PLACEHOLDERS (44561, 60478, 43897) INTO MEANINGFUL HEADERS OR VALUES.

MAP ALL FINANCIAL DATA FROM THE TABLES AND KEY-VALUE PAIRS TO THE MATRIX.

### NOTE : THE TABLES HAVE BEEN EXTRACTED FROM IMAGES HENCE TEABLE HEADER MIGHT BE HERE AND THERE LIKE ALLIGNMENT ISSUE DON'T BLINDLY ADD VALUES UNDETAND THE CONETXT

DO NOT HALLUCINATE ANY DATA OR MAKE UP ANY VALUES.

ONLY FILL DATA ACCORDING TO THE CONTEXT PROVIDED.

I NEED THE MATRIX IN THE EXACT SAME STRUCTURE AS RECEIVED — DO NOT CHANGE FORMATTING OR LAYOUT. ONLY FILL IN THE MISSING CELLS ACCURATELY.

RETURN YOUR RESPONSE AS A JSON OBJECT CONTAINING THE FILLED MATRIX AND STATISTICS.

CONTEXT INFORMATION FROM FINANCIAL DOCUMENT:
----------------------------------------

{context}"""

    logger.info("Making API call to OpenAI")
    
    try:
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using a current model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=16000,
            response_format={"type": "json_object"}  # Force JSON response format
        )
        
        # Save the raw API response and processed content
        raw_response = response
        content = response.choices[0].message['content'].strip()
        
        # Save debugging information to a file
        debug_data = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "financial_data_summary": {
                "tables_count": len(financial_data.get('tables', [])),
                "key_value_pairs_count": len(financial_data.get('key_value_pairs', [])),
                "metadata": financial_data.get('metadata', {})
            },
            "matrix_shape": f"{len(normalized_matrix)}x{len(normalized_matrix[0]) if normalized_matrix else 0}",
            "raw_api_response": str(raw_response),  # Include full API response
            "response_content": content
        }
        
        debug_file = f"mapping_debug_{int(time.time())}.json"
        try:
            with open(debug_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
            logger.info(f"Debug data saved to {debug_file}")
        except Exception as debug_err:
            logger.warning(f"Could not save debug data: {debug_err}")
        
        # Parse the JSON response
        parsed_content = json.loads(content)
        
        # Extract the filled matrix
        if "Filled Matrix" in parsed_content:
            filled_matrix = parsed_content["Filled Matrix"]
            
            # Create initial result object
            initial_result = {
                "filled_matrix": filled_matrix,
                "original_matrix": matrix,
                "stats": {
                    "rows_filled": parsed_content.get("Number of rows filled", 0),
                    "columns_filled": parsed_content.get("Number of columns filled", 0),
                    "cells_filled": parsed_content.get("Number of cells filled", 0),
                    "cells_not_filled": parsed_content.get("Number of cells not filled", 0)
                },
                "debug_file": debug_file
            }
            
            logger.info(f"Successfully filled matrix: {initial_result['stats']['cells_filled']} cells filled")
            
            # VALIDATION STEP - Automatically validate the filled matrix
            logger.info("Starting validation step...")
            validation_result = validate_filled_matrix(filled_matrix, financial_data, context)
            
            final_matrix = filled_matrix
            cells_changed = 0
            
            # If validation found issues and provided a corrected matrix, use it
            if validation_result.get('is_matrix_correct') == "No" and "corrected_matrix" in validation_result:
                final_matrix = validation_result["corrected_matrix"]
                cells_changed = validation_result.get('cells_changed', 0)
                logger.info(f"Validation changed {cells_changed} cells in the matrix")
            
            # Create final result with validation data
            result = {
                "filled_matrix": final_matrix,
                "original_matrix": matrix,
                "stats": {
                    "rows_filled": parsed_content.get("Number of rows filled", 0),
                    "columns_filled": parsed_content.get("Number of columns filled", 0),
                    "cells_filled": parsed_content.get("Number of cells filled", 0),
                    "cells_not_filled": parsed_content.get("Number of cells not filled", 0)
                },
                "validation": {
                    "is_matrix_correct": validation_result.get('is_matrix_correct', "Unknown"),
                    "cells_changed": cells_changed,
                    "validation_notes": validation_result.get('validation_notes', ""),
                    "debug_file": validation_result.get('debug_file')
                },
                "debug_file": debug_file
            }
            
            return result
            
        else:
            logger.warning("No filled matrix found in response")
            return {
                "error": "No filled matrix found in API response",
                "raw_response": parsed_content,
                "debug_file": debug_file
            }
    
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {str(e)}")
        return {
            "error": f"Error calling OpenAI API: {str(e)}"
        }

def save_context_to_file(financial_data, output_path):
    """
    Save the formatted context to a file for debugging
    
    Args:
        financial_data (dict): Extracted financial data
        output_path (str): Path to save the context
        
    Returns:
        bool: Success status
    """
    try:
        context = format_pdf_data_as_context(financial_data)
        with open(output_path, 'w') as f:
            f.write(context)
        logger.info(f"Saved formatted context to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving context to file: {e}")
        return False

# Add a function to start a query session after mapping
def start_query_session(financial_data, filled_matrix, sheet_url=None, sheet_name=None, spreadsheet_id=None):
    """
    Start an interactive query session with the mapped financial data
    
    Args:
        financial_data (dict): Original financial data extracted from PDF
        filled_matrix (list): Matrix filled with financial data
        sheet_url (str, optional): URL of the connected Google Sheet
        sheet_name (str, optional): Name of the sheet containing the data
        spreadsheet_id (str, optional): ID of the Google Spreadsheet
        
    Returns:
        None
    """
    # Import the sheets_manager module only when needed
    try:
        from sheets_manager import extract_spreadsheet_id, get_sheet_data, update_sheet_data
        sheets_integration = True
    except ImportError:
        logger.warning("sheets_manager module not found. Google Sheets functionality will be limited.")
        sheets_integration = False
    
    # Make sure we have a valid spreadsheet ID
    if sheet_url and not spreadsheet_id:
        try:
            from sheets_manager import extract_spreadsheet_id
            spreadsheet_id = extract_spreadsheet_id(sheet_url)
        except:
            logger.warning("Could not extract spreadsheet ID from URL.")
    
    # Set up the query processor
    print("\n=== Financial Data Query Agent ===")
    print("Ask questions about your financial data, request sheet operations, or analyze the data.")
    print("Type 'exit' to end the session.\n")
    
    print("Financial data loaded:")
    print(f"- {len(financial_data.get('tables', []))} tables")
    print(f"- {len(financial_data.get('key_value_pairs', []))} key-value pairs")
    if filled_matrix:
        print(f"- Matrix dimensions: {len(filled_matrix)}x{len(filled_matrix[0]) if filled_matrix and len(filled_matrix) > 0 else 0}")
    if spreadsheet_id:
        print(f"- Connected to Google Sheet: {spreadsheet_id}")
    if sheet_name:
        print(f"- Active sheet: {sheet_name}")
    print("\n")
    print("Example queries:")
    print("- What is the total revenue for 2022?")
    print("- What are the main expense categories?")
    print("- Update cell B5 to $250,000")
    print("- Calculate the profit margin")
    print("- Find the biggest expense item")
    print("\n")
    
    # Initialize conversation history for context
    messages = [
        {
            "role": "system",
            "content": f"""You are a financial data expert who helps users analyze and understand their financial data.
You have access to financial data that has been mapped to a structured matrix. 

The data includes:
- Financial information extracted from a PDF document
- A structured matrix with {len(filled_matrix)}x{len(filled_matrix[0]) if filled_matrix and len(filled_matrix) > 0 else 0} dimensions
{f"- Connection to a Google Sheet with ID: {spreadsheet_id}" if spreadsheet_id else ""}
{f"- Active sheet: {sheet_name}" if sheet_name else ""}

You can:
1. Answer questions about the financial data
2. Explain trends and insights
3. Provide analysis of financial metrics
4. Help update values in the sheet
5. Perform calculations

When the user asks you to update the sheet, respond with:
ACTION: update_sheet
PARAMETERS: {{"cell": "B5", "value": "250000"}}
EXPLANATION: I'll update cell B5 to the value 250000.

Always answer concisely and focus on providing accurate financial insights."""
        }
    ]
    
    # Process user queries
    while True:
        try:
            # Get user input
            user_query = input("\nYou: ")
            
            # Check for exit command
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("\nEnding session. Goodbye!")
                break
            
            # Add user message to context
            messages.append({"role": "user", "content": user_query})
            
            # Call OpenAI API with the full context
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
                break
            
            openai.api_key = api_key
            
            # Call the API
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,  # More creative for conversation
                max_tokens=8000
            )
            
            # Get the response
            content = response.choices[0].message['content'].strip()
            
            # Add to conversation history
            messages.append({"role": "assistant", "content": content})
            
            # Check for action requests
            action_match = re.search(r'ACTION:\s*(\w+)', content)
            if action_match:
                action = action_match.group(1)
                
                # Extract parameters
                params_match = re.search(r'PARAMETERS:\s*({.*})', content, re.DOTALL)
                params = {}
                if params_match:
                    try:
                        params = json.loads(params_match.group(1))
                    except:
                        print("Warning: Could not parse action parameters.")
                
                # Handle update_sheet action
                if action == "update_sheet" and sheets_integration:
                    if not spreadsheet_id:
                        print("Error: No Google Sheet connected.")
                    else:
                        try:
                            cell = params.get("cell")
                            value = params.get("value")
                            active_sheet = params.get("sheet_name", sheet_name or "Sheet1")
                            
                            # Convert cell reference to row/column
                            col_letter = ''.join(c for c in cell if c.isalpha()).upper()
                            row_num = int(''.join(c for c in cell if c.isdigit())) - 1  # 0-based
                            
                            # Convert column letter to index
                            col_num = 0
                            for i, char in enumerate(reversed(col_letter)):
                                col_num += (ord(char) - ord('A') + 1) * (26 ** i)
                            col_num -= 1  # 0-based
                            
                            # Get the current sheet data
                            sheet_data = get_sheet_data(spreadsheet_id, active_sheet)
                            
                            # Ensure the matrix is large enough
                            while len(sheet_data) <= row_num:
                                sheet_data.append([])
                            for i in range(len(sheet_data)):
                                while len(sheet_data[i]) <= col_num:
                                    sheet_data[i].append("")
                            
                            # Update the cell
                            sheet_data[row_num][col_num] = value
                            
                            # Update the sheet
                            success = update_sheet_data(spreadsheet_id, active_sheet, sheet_data)
                            
                            if success:
                                print(f"✅ Successfully updated cell {cell} to {value} in sheet '{active_sheet}'")
                                
                                # Also update the filled matrix to keep it in sync
                                if filled_matrix:
                                    while len(filled_matrix) <= row_num:
                                        filled_matrix.append([])
                                    for i in range(len(filled_matrix)):
                                        while len(filled_matrix[i]) <= col_num:
                                            filled_matrix[i].append("")
                                    filled_matrix[row_num][col_num] = value
                            else:
                                print(f"❌ Failed to update cell {cell} in sheet '{active_sheet}'")
                        except Exception as e:
                            print(f"Error updating sheet: {str(e)}")
                elif action == "update_sheet" and not sheets_integration:
                    print("❌ Google Sheets integration is not available.")
            
            # Print the response without the action part
            if action_match:
                explanation_match = re.search(r'EXPLANATION:(.*?)(?:ACTION:|$)', content, re.DOTALL)
                if explanation_match:
                    print("\nAgent:", explanation_match.group(1).strip())
                else:
                    # If no explanation, remove the action block and print the rest
                    action_block = re.search(r'ACTION:.*?(?=\n\n|$)', content, re.DOTALL)
                    if action_block:
                        display_content = content.replace(action_block.group(0), "").strip()
                        print("\nAgent:", display_content)
                    else:
                        print("\nAgent:", content)
            else:
                print("\nAgent:", content)
        
        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Let's continue with a new query.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Map financial data to a matrix using OpenAI")
    parser.add_argument("--data-file", required=True, help="Path to the financial data JSON file")
    parser.add_argument("--matrix-file", required=True, help="Path to the matrix template JSON file")
    parser.add_argument("--output", required=True, help="Path to save the results")
    parser.add_argument("--save-context", help="Path to save the formatted context (optional)")
    parser.add_argument("--sheet-url", help="URL of Google Sheet to connect to (optional)")
    parser.add_argument("--sheet-name", help="Name of the sheet to use (optional)")
    parser.add_argument("--interactive", action="store_true", help="Start interactive query session after mapping")
    
    args = parser.parse_args()
    
    try:
        # Load financial data
        with open(args.data_file, 'r') as f:
            financial_data = json.load(f)
        
        # Load matrix template
        with open(args.matrix_file, 'r') as f:
            matrix = json.load(f)
        
        # Save context if requested
        if args.save_context:
            save_context_to_file(financial_data, args.save_context)
        
        # Map financial data to matrix
        result = map_financial_data(financial_data, matrix)
        
        # Save result
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
        
        # Start interactive session if requested
        if args.interactive:
            filled_matrix = result.get("filled_matrix")
            if filled_matrix:
                # Extract spreadsheet ID if provided
                spreadsheet_id = None
                if args.sheet_url:
                    try:
                        from sheets_manager import extract_spreadsheet_id
                        spreadsheet_id = extract_spreadsheet_id(args.sheet_url)
                    except:
                        pass
                
                start_query_session(
                    financial_data=financial_data,
                    filled_matrix=filled_matrix,
                    sheet_url=args.sheet_url,
                    sheet_name=args.sheet_name,
                    spreadsheet_id=spreadsheet_id
                )
            else:
                logger.error("No filled matrix available for interactive session.")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1) 