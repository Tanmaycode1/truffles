import json
import re
from mapping_engine import process_query, get_best_match, calculate_similarity
from sheets_manager import update_sheet_with_data
from pdf_processor import update_sheet_data, get_sheet_data, get_sheet_names
import openai
import os
import logging
import time
import uuid
from dotenv import load_dotenv
import traceback

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load environment variables
load_dotenv()

class QuerySession:
    """Class to manage an ongoing query session with context memory"""
    
    def __init__(self, financial_data=None, filled_matrix=None, sheet_url=None, sheet_name=None, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.financial_data = financial_data
        self.filled_matrix = filled_matrix  # Keep for backward compatibility
        self.sheet_matrices = {}  # Store matrices for each sheet by name
        self.sheet_url = sheet_url
        self.sheet_name = sheet_name
        self.message_history = []
        self.available_sheets = []  # Store available sheet names
        
        # Initialize the available matrices from all_mapping_results if filled_matrix is not None
        if filled_matrix is not None:
            self.sheet_matrices["default"] = filled_matrix
            
        # Initialize available sheets if sheet_url is provided
        if self.sheet_url:
            self._load_available_sheets()
            
        self.context = self._build_initial_context()
        
    def _load_available_sheets(self):
        """
        Load available sheets from the Google Sheet and store their matrices
        """
        try:
            # Try different import paths
            try:
                from google_sheets_access import get_sheet_data, get_worksheet_names
            except ModuleNotFoundError:
                try:
                    from sheets_manager import get_sheet_data, get_worksheet_names
                except ModuleNotFoundError:
                    from truffles.sheets_manager import get_sheet_data, get_worksheet_names
                    
            # Get available worksheet names
            self.available_sheets = get_worksheet_names(self.sheet_url)
            logger.info(f"Found sheets: {self.available_sheets}")
            
            # Load data for each available sheet
            for sheet_name in self.available_sheets:
                try:
                    sheet_data = get_sheet_data(self.sheet_url, sheet_name)
                    if sheet_data:
                        self.sheet_matrices[sheet_name] = sheet_data
                        logger.info(f"Loaded matrix for sheet '{sheet_name}': {len(sheet_data)}x{len(sheet_data[0]) if sheet_data and len(sheet_data) > 0 else 0}")
                except Exception as e:
                    logger.warning(f"Failed to load matrix for sheet '{sheet_name}': {str(e)}")
            
            # If no filled matrix is set and we have sheets, use the first one as default
            if not self.filled_matrix and self.sheet_matrices:
                self.filled_matrix = next(iter(self.sheet_matrices.values()))
                
        except Exception as e:
            logger.error(f"Failed to load available sheets: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Continue with empty lists as fallback
            self.available_sheets = []
        
    def _build_initial_context(self):
        """Build the initial context based on available data"""
        context_parts = []
        
        if self.financial_data:
            context_parts.append("=== FINANCIAL DATA AVAILABLE ===")
            if self.financial_data.get('metadata'):
                doc_type = self.financial_data.get('metadata', {}).get('document_type', 'Unknown')
                context_parts.append(f"DOCUMENT TYPE: {doc_type}")
                
                company_name = self.financial_data.get('metadata', {}).get('company_name', 'Unknown')
                if company_name != 'Unknown':
                    context_parts.append(f"COMPANY: {company_name}")
                
                statement_date = self.financial_data.get('metadata', {}).get('statement_date', 'Unknown')
                if statement_date != 'Unknown':
                    context_parts.append(f"STATEMENT DATE: {statement_date}")
            
            # Add key financial metrics 
            context_parts.append("\n=== KEY FINANCIAL METRICS ===")
            metrics_found = False
            
            # Look for important financial metrics in key-value pairs
            financial_keywords = ["revenue", "income", "profit", "sales", "earnings", "ebitda", 
                                 "assets", "liabilities", "equity", "cash", "expense", "total"]
            
            metric_pairs = []
            for kv in self.financial_data.get('key_value_pairs', []):
                key = kv.get('key', '').lower()
                value = kv.get('value', '')
                
                # Check if this key contains financial keywords
                if any(keyword in key for keyword in financial_keywords):
                    metric_pairs.append(f"{kv.get('key')}: {value}")
                    metrics_found = True
            
            if metric_pairs:
                context_parts.extend(metric_pairs[:10])  # Limit to 10 metrics
            
            if not metrics_found:
                context_parts.append("No specific financial metrics found in key-value pairs.")
            
            # Add table summaries
            num_tables = len(self.financial_data.get('tables', []))
            if num_tables > 0:
                context_parts.append(f"\n=== TABLES SUMMARY ({num_tables} tables) ===")
                for i, table in enumerate(self.financial_data.get('tables', [])[:3]):  # Limit to 3 tables
                    table_type = table.get('type', f'Table {i+1}')
                    headers = table.get('headers', [])
                    rows = len(table.get('data', []))
                    
                    context_parts.append(f"TABLE {i+1}: {table_type} ({rows} rows)")
                    if headers:
                        context_parts.append(f"Headers: {', '.join(headers[:5])}" + ("..." if len(headers) > 5 else ""))
                    
                    # Add a sample of data from the first few rows if available
                    data = table.get('data', [])
                    if data and len(data) > 0:
                        context_parts.append("Sample data:")
                        for row_idx, row in enumerate(data[:3]):  # Show first 3 rows
                            row_str = " | ".join([str(cell) for cell in row[:5]])
                            if len(row) > 5:
                                row_str += " | ..."
                            context_parts.append(f"  Row {row_idx+1}: {row_str}")
            
            # Add key-value pairs summary
            num_kv_pairs = len(self.financial_data.get('key_value_pairs', []))
            context_parts.append(f"\n=== KEY-VALUE PAIRS SUMMARY ({num_kv_pairs} pairs) ===")
        else:
            context_parts.append("FINANCIAL DATA AVAILABLE: No")
        
        # Add matrices information
        if self.sheet_matrices:
            context_parts.append(f"\n=== AVAILABLE MATRICES ===")
            for sheet_name, matrix in self.sheet_matrices.items():
                # Skip preview matrices for context
                if sheet_name.endswith("_preview"):
                    continue
                    
                rows = len(matrix) if matrix else 0
                cols = len(matrix[0]) if matrix and rows > 0 else 0
                context_parts.append(f"Matrix for '{sheet_name}': {rows}x{cols}")
        
        if self.filled_matrix:
            filled_rows = len(self.filled_matrix)
            filled_cols = len(self.filled_matrix[0]) if filled_rows > 0 else 0
            context_parts.append(f"\n=== FILLED MATRIX ({filled_rows}x{filled_cols}) ===")
            
            # Add matrix headers (first row) if available
            if filled_rows > 0:
                headers = self.filled_matrix[0]
                headers_str = " | ".join([str(h) for h in headers[:5]])
                if filled_cols > 5:
                    headers_str += " | ..."
                context_parts.append(f"Headers: {headers_str}")
                
                # Add first few data rows
                for row_idx, row in enumerate(self.filled_matrix[1:6]):  # Show up to 5 data rows
                    row_str = " | ".join([str(cell) for cell in row[:5]])
                    if filled_cols > 5:
                        row_str += " | ..."
                    context_parts.append(f"Row {row_idx+1}: {row_str}")
                
                # Look for revenue or profit rows
                revenue_rows = []
                profit_rows = []
                for row_idx, row in enumerate(self.filled_matrix):
                    # Check if first cell (usually contains labels) has keywords
                    if row and len(row) > 0 and isinstance(row[0], str):
                        first_cell = row[0].lower()
                        if any(term in first_cell for term in ["revenue", "sales", "income"]):
                            revenue_str = " | ".join([str(cell) for cell in row[:5]])
                            if filled_cols > 5:
                                revenue_str += " | ..."
                            revenue_rows.append(f"Revenue ({row_idx+1}): {revenue_str}")
                        elif any(term in first_cell for term in ["profit", "earnings", "ebit"]):
                            profit_str = " | ".join([str(cell) for cell in row[:5]])
                            if filled_cols > 5:
                                profit_str += " | ..."
                            profit_rows.append(f"Profit ({row_idx+1}): {profit_str}")
                
                if revenue_rows:
                    context_parts.append("\nRevenue Information:")
                    context_parts.extend(revenue_rows[:3])  # Limit to 3 revenue rows
                
                if profit_rows:
                    context_parts.append("\nProfit Information:")
                    context_parts.extend(profit_rows[:3])  # Limit to 3 profit rows
        else:
            context_parts.append("FILLED MATRIX: Not available")
        
        if self.sheet_url:
            context_parts.append(f"\n=== GOOGLE SHEET ===")
            context_parts.append(f"URL: {self.sheet_url}")
            
            # Add available sheets information
            if self.available_sheets:
                context_parts.append(f"AVAILABLE SHEETS: {', '.join(self.available_sheets)}")
                
                # Add preview of each sheet (first 2 rows only)
                context_parts.append("\nSHEET PREVIEWS:")
                for sheet_name, matrix in self.sheet_matrices.items():
                    # Skip matrices that aren't previews
                    if not sheet_name.endswith("_preview") and sheet_name != "default":
                        continue
                        
                    if matrix and len(matrix) > 0:
                        display_name = sheet_name.replace("_preview", "") if sheet_name.endswith("_preview") else sheet_name
                        context_parts.append(f"\n{display_name} (Preview):")
                        for row_idx, row in enumerate(matrix[:2]):  # First 2 rows only
                            row_str = " | ".join([str(cell) for cell in row[:5]])
                            if len(row) > 5:
                                row_str += " | ..."
                            context_parts.append(f"  Row {row_idx+1}: {row_str}")
            
            if self.sheet_name:
                context_parts.append(f"ACTIVE SHEET: {self.sheet_name}")
        else:
            context_parts.append("\nGOOGLE SHEET: Not connected")
            
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
        
        # Return the combined messages
        return [system_message] + self.message_history
    
    def _get_system_prompt(self):
        """Get the system prompt with full context"""
        return f"""You are a financial data expert that helps users analyze and understand their financial data.
You have access to financial data that has been extracted from a PDF document and mapped to a structured matrix.

DETAILED CONTEXT:
{self.context}

You should use this financial data to:
1. Answer specific questions about financial metrics (revenue, profit, expenses, etc.)
2. Provide calculations and analysis when requested
3. Explain financial trends and patterns
4. Compare values across different time periods or categories
5. Make specific updates to the Google Sheet when requested

Important instructions for handling Google Sheet updates:
- CRITICAL: When a user asks to update a specific CELL (like B5), ALWAYS use update_sheet action
- CRITICAL: When a user asks to update a named VALUE (like revenue or EBITDA), ALWAYS use update_named_value action
- Do not use find_cell for any update requests, it is only for informational queries
- Always provide the sheet name parameter in your action
- Be flexible with sheet names - users might use variations like "income" instead of "Income Statement"
- If you're having trouble accessing sheets, use the debug_sheets action to troubleshoot
- If a user asks to regenerate or remap data, use the regenerate_mapping action

UPDATE OPERATIONS USE THESE EXACT FORMATS:

Direct cell update (like "Update cell B5 to 250000 in income sheet"):
ACTION: update_sheet
PARAMETERS: {{"cell": "B5", "value": "250000", "sheet_name": "Income Statement"}}
EXPLANATION: I'll update cell B5 to the value 250000 in the Income Statement sheet.

Named value update (like "Update EBITDA to 30000" or "Change revenue to 500000"):
ACTION: update_named_value
PARAMETERS: {{"value_name": "EBITDA", "value": "30000", "sheet_name": "Income Statement"}}
EXPLANATION: I'll update the EBITDA value to 30000 in the Income Statement sheet.

Regenerate mapping (like "Regenerate the mapping" or "Remap the data"):
ACTION: regenerate_mapping
PARAMETERS: {{}}
EXPLANATION: I'll regenerate the complete mapping for all sheets using the current financial data.

For migrations and other operations:
ACTION: migrate_matrix
PARAMETERS: {{"sheet_name": "Income Statement", "create_new": false, "source_sheet": "Income Statement"}}
EXPLANATION: I'll migrate the Income Statement matrix to the Income Statement sheet in Google Sheets.

ACTION: migrate_all_matrices
PARAMETERS: {{}}
EXPLANATION: I'll migrate all financial statements to their matching sheets in Google Sheets.

ACTION: debug_sheets
PARAMETERS: {{}}
EXPLANATION: Let me diagnose the Google Sheets connection to find any issues.

This allows the system to execute the requested spreadsheet operation."""

    def to_dict(self):
        """Convert session to a dictionary for storage"""
        return {
            "session_id": self.session_id,
            "context": self.context,
            "message_history": self.message_history,
            "available_sheets": self.available_sheets,
            # Only store previews of sheet matrices to keep size manageable
            "sheet_matrices_preview": {name: matrix[:2] if matrix and len(matrix) > 0 else [] 
                                      for name, matrix in self.sheet_matrices.items()},
            # Don't include the large data objects, just references
            "has_financial_data": self.financial_data is not None,
            "has_filled_matrix": self.filled_matrix is not None,
            "sheet_url": self.sheet_url,
            "sheet_name": self.sheet_name
        }
    
    @classmethod
    def from_dict(cls, data, financial_data=None, filled_matrix=None):
        """Create a session from a dictionary"""
        session = cls(
            financial_data=financial_data,
            filled_matrix=filled_matrix,
            sheet_url=data.get("sheet_url"),
            sheet_name=data.get("sheet_name"),
            session_id=data.get("session_id")
        )
        session.message_history = data.get("message_history", [])
        
        # Initialize these in case they weren't set during __init__
        if "available_sheets" in data:
            session.available_sheets = data.get("available_sheets", [])
        if "sheet_matrices_preview" in data:
            session.sheet_matrices = data.get("sheet_matrices_preview", {})
            
        return session

    def get_matrix_for_sheet(self, sheet_name):
        """
        Get the matrix for a specific sheet by name
        Returns None if no matrix is found for that sheet
        """
        # First try exact match
        if sheet_name in self.sheet_matrices:
            return self.sheet_matrices[sheet_name]
            
        # Try case-insensitive match
        sheet_name_lower = sheet_name.lower()
        for name, matrix in self.sheet_matrices.items():
            if name.lower() == sheet_name_lower:
                return matrix
                
        # Try more careful partial matching for financial statements
        # This helps distinguish between "Balance Sheet", "Income Statement", etc.
        financial_statement_types = [
            ("balance sheet", ["balance", "balance sheet", "assets", "liabilities"]),
            ("income statement", ["income", "income statement", "profit loss", "p&l", "earnings"]),
            ("cash flow", ["cash flow", "statement of cash", "cash flows"])
        ]
        
        for name, matrix in self.sheet_matrices.items():
            name_lower = name.lower()
            sheet_type = None
            
            # Determine the type of the requested sheet
            for fs_type, keywords in financial_statement_types:
                if any(keyword in sheet_name_lower for keyword in keywords):
                    sheet_type = fs_type
                    break
                    
            # If we identified a type for the requested sheet, make sure it matches the matrix type
            if sheet_type:
                for fs_type, keywords in financial_statement_types:
                    if fs_type == sheet_type and any(keyword in name_lower for keyword in keywords):
                        return matrix
        
        # Only now try general partial match if we couldn't find a match by statement type
        for name, matrix in self.sheet_matrices.items():
            if sheet_name_lower in name.lower() or name.lower() in sheet_name_lower:
                return matrix
                
        # If we have a default filled matrix and no specific match was found, 
        # return that as a fallback
        if self.filled_matrix and not self.sheet_matrices:
            return self.filled_matrix
            
        # No match found
        return None

def process_natural_language_query(query, pdf_data, sheet_structure, sheet_url=None):
    """
    Process a natural language query about the PDF data and sheet structure
    
    Args:
        query (str): The natural language query
        pdf_data (dict): Extracted data from PDF
        sheet_structure (dict): Structure of Google Sheet
        sheet_url (str, optional): URL of the Google Sheet
        
    Returns:
        dict: Query result
    """
    # Process the query using the mapping engine's process_query function
    query_result = process_query(query, pdf_data, sheet_structure)
    
    # If the query wasn't well-handled by rule-based processing, try using GPT
    if query_result.get('query_type') == 'unknown' and os.getenv('OPENAI_API_KEY'):
        try:
            query_result = process_query_with_gpt(query, pdf_data, sheet_structure)
        except Exception as e:
            query_result['error'] = str(e)
    
    # Check if this is an update request and we need to actually update the sheet
    if query_result.get('query_type') == 'update_request' and query_result.get('update') and sheet_url:
        try:
            # Format the update in the expected structure
            mapping_result = {
                'updates': [query_result['update']]
            }
            
            # Update the sheet
            update_result = update_sheet_with_data(sheet_url, mapping_result)
            
            # Add the update result to the query result
            query_result['update_result'] = update_result
            
        except Exception as e:
            query_result['error'] = str(e)
    
    return query_result

def process_query_with_gpt(query, pdf_data, sheet_structure):
    """
    Process a natural language query using OpenAI's GPT model
    
    Args:
        query (str): The natural language query
        pdf_data (dict): Extracted data from PDF
        sheet_structure (dict): Structure of Google Sheet
        
    Returns:
        dict: Query result
    """
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Prepare context for the model
    # Get PDF summary
    pdf_summary = get_pdf_summary(pdf_data)
    
    # Get sheet summary
    sheet_summary = get_sheet_summary(sheet_structure)
    
    # Get relevant data from PDF based on query
    relevant_data = search_pdf_data(pdf_data, query)
    
    # Construct prompt
    prompt = f"""
You are an AI assistant that helps with financial data analysis.

FINANCIAL DOCUMENT SUMMARY:
Document type: {pdf_summary.get('document_type', 'Unknown')}
Date: {pdf_summary.get('statement_date', 'Unknown')}
Key metrics: {', '.join([f"{m['name']}: {m['value']}" for m in pdf_summary.get('key_metrics', [])][:5])}

SPREADSHEET STRUCTURE:
Title: {sheet_summary.get('title', 'Unknown')}
Sheets: {', '.join([s['title'] for s in sheet_summary.get('sheets', [])])}

USER QUERY:
{query}

Based on this information, please analyze and respond to the query. If it's a request to update the spreadsheet, specify which cell should be updated with what value.
"""
    
    # Call OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI assistant that helps with financial data analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1200,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    
    # Parse the response
    gpt_response = response.choices[0].message['content'].strip()
    
    # Check if the response indicates an update request
    update_pattern = r'update|fill|put|set|write|place|add'
    if re.search(update_pattern, query.lower()) and re.search(r'cell|row|column|field', query.lower()):
        # This appears to be an update request
        # Try to extract cell and value information
        cell_match = re.search(r'cell[s]?\s+([A-Z]+\d+)', gpt_response, re.IGNORECASE)
        value_match = re.search(r'value[s]?[\s:]+([0-9,.()-]+)', gpt_response, re.IGNORECASE)
        sheet_match = re.search(r'sheet[s]?[\s:]+[\'"]*([A-Za-z0-9 ]+)[\'"]*', gpt_response, re.IGNORECASE)
        
        if cell_match and value_match:
            cell = cell_match.group(1)
            value = value_match.group(1).strip()
            sheet_name = sheet_match.group(1) if sheet_match else sheet_structure['sheets'][0]['title']
            
            # Clean value for numeric processing
            clean_value = value.replace(',', '')
            if '(' in clean_value and ')' in clean_value:
                clean_value = clean_value.replace('(', '-').replace(')', '')
                
            try:
                numeric_value = float(clean_value)
            except ValueError:
                numeric_value = None
            
            return {
                'query_type': 'update_request',
                'update': {
                    'sheet_name': sheet_name,
                    'range': cell,
                    'value': numeric_value if numeric_value is not None else value
                },
                'response': f"Will update cell {cell} in sheet '{sheet_name}' with value {value}"
            }
    
    # If not an update request, return the GPT response
    return {
        'query_type': 'gpt_response',
        'response': gpt_response
    }

def get_pdf_summary(pdf_data):
    """
    Generate a summary of the PDF data
    
    Args:
        pdf_data (dict): Extracted data from PDF
        
    Returns:
        dict: Summary of PDF data
    """
    summary = {
        'page_count': pdf_data.get('page_count', 0),
        'tables_count': len(pdf_data.get('tables', [])),
        'key_value_pairs_count': len(pdf_data.get('key_value_pairs', [])),
        'document_type': 'Unknown',
        'statement_date': 'Unknown',
        'key_metrics': [],
        'extraction_methods': pdf_data.get('processing_methods', [])
    }
    
    # Find document type and date
    for kv in pdf_data.get('key_value_pairs', []):
        if kv.get('key') == 'document_type':
            summary['document_type'] = kv.get('value', 'Unknown')
        elif kv.get('key') == 'statement_date':
            summary['statement_date'] = kv.get('value', 'Unknown')
    
    # Find key financial metrics based on common terms or importance flag
    for kv in pdf_data.get('key_value_pairs', []):
        key = kv.get('key', '').lower()
        value = kv.get('value', '')
        
        # Check if this is marked as important
        is_important = kv.get('important', False)
        
        # Check for key financial terms
        is_key_term = any(term in key for term in [
            'total assets', 'total liabilities', 'equity', 'net income', 
            'revenue', 'profit', 'earnings', 'cash', 'total'
        ])
        
        if (is_important or is_key_term) and kv.get('numeric_value') is not None:
            summary['key_metrics'].append({
                'name': kv.get('key'),
                'value': value,
                'numeric_value': kv.get('numeric_value')
            })
    
    # Sort key metrics by name to get a consistent order
    summary['key_metrics'].sort(key=lambda x: x['name'])
    
    # Use grouped data if available
    if 'grouped_data' in pdf_data:
        for category, items in pdf_data['grouped_data'].items():
            if category != 'other' and category != 'metadata':
                summary[f'{category}_items_count'] = len(items)
    
    return summary

def get_sheet_summary(sheet_structure):
    """
    Generate a summary of the Google Sheet structure
    
    Args:
        sheet_structure (dict): Structure of Google Sheet
        
    Returns:
        dict: Summary of sheet structure
    """
    summary = {
        'title': sheet_structure.get('title', 'Unknown'),
        'sheets': []
    }
    
    for sheet in sheet_structure.get('sheets', []):
        sheet_summary = {
            'title': sheet.get('title', ''),
            'rows': sheet.get('rows', 0),
            'cols': sheet.get('cols', 0),
            'appears_transposed': sheet.get('appears_transposed', False),
            'is_financial_statement': sheet.get('is_financial_statement', False),
            'headers': sheet.get('headers', [])[:5],  # Get first 5 headers only
            'row_headers': sheet.get('rows_as_headers', [])[:5]  # Get first 5 row headers only
        }
        
        summary['sheets'].append(sheet_summary)
    
    return summary

def search_pdf_data(pdf_data, search_term):
    """
    Search for a term in the PDF data
    
    Args:
        pdf_data (dict): Extracted data from PDF
        search_term (str): Term to search for
        
    Returns:
        list: Matching items
    """
    matches = []
    
    # Get words from search term for partial matching
    search_words = re.findall(r'\b\w+\b', search_term.lower())
    
    # Search in key-value pairs
    for kv in pdf_data.get('key_value_pairs', []):
        key = kv.get('key', '')
        value = kv.get('value', '')
        
        # Check for both exact similarity and word containment
        key_similarity = calculate_similarity(search_term, key)
        key_word_match = any(word in key.lower() for word in search_words)
        
        if key_similarity > 0.7 or key_word_match:
            matches.append({
                'type': 'key_value_pair',
                'key': key,
                'value': value,
                'similarity': key_similarity if key_similarity > 0.7 else 0.5,
                'page': kv.get('page', 1),
                'extraction_method': kv.get('extraction_method', 'unknown')
            })
    
    # Search in tables
    for table in pdf_data.get('tables', []):
        table_name = table.get('table_name', '')
        headers = table.get('headers', [])
        data = table.get('data', [])
        
        # Check table headers
        for header in headers:
            if not header or not isinstance(header, str):
                continue
                
            header_similarity = calculate_similarity(search_term, header)
            header_word_match = any(word in header.lower() for word in search_words)
            
            if header_similarity > 0.7 or header_word_match:
                matches.append({
                    'type': 'table_header',
                    'table': table_name,
                    'header': header,
                    'similarity': header_similarity if header_similarity > 0.7 else 0.5,
                    'extraction_method': table.get('extraction_method', 'unknown')
                })
        
        # Check table data (first column often contains row labels)
        for row in data:
            if not row or len(row) == 0:
                continue
                
            row_label = row[0]
            if not row_label or not isinstance(row_label, str):
                continue
                
            row_similarity = calculate_similarity(search_term, row_label)
            row_word_match = any(word in row_label.lower() for word in search_words)
            
            if row_similarity > 0.7 or row_word_match:
                # Get the values in this row
                row_values = row[1:] if len(row) > 1 else []
                
                matches.append({
                    'type': 'table_row',
                    'table': table_name,
                    'row_label': row_label,
                    'values': row_values,
                    'similarity': row_similarity if row_similarity > 0.7 else 0.5,
                    'extraction_method': table.get('extraction_method', 'unknown')
                })
    
    # Check grouped data if available
    if 'grouped_data' in pdf_data:
        # Check if any categories match the search term
        for category, items in pdf_data['grouped_data'].items():
            if category == 'other' or category == 'metadata':
                continue
                
            category_similarity = calculate_similarity(search_term, category.replace('_', ' '))
            category_word_match = any(word in category.lower() for word in search_words)
            
            if category_similarity > 0.7 or category_word_match:
                matches.append({
                    'type': 'category',
                    'category': category,
                    'items_count': len(items),
                    'similarity': category_similarity if category_similarity > 0.7 else 0.5
                })
    
    # Sort matches by similarity
    matches.sort(key=lambda x: x['similarity'], reverse=True)
    
    return matches 

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
        # Pre-process query to determine if it's an update request and route appropriately
        query_lower = query.lower()
        is_update_request = any(term in query_lower for term in ["update", "change", "set", "modify"])
        
        # Add the user's message to the history
        session.add_message("user", query)
        
        # Call OpenAI with the full conversation history
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
        
        openai.api_key = api_key
        
        # Get all messages for the API call
        messages = session.get_messages_for_api()
        
        # If this is a known update request, add a specific system instruction
        # to ensure the model uses the right action
        if is_update_request:
            # Analyze if this is a cell update or named value update
            is_cell_update = re.search(r'\bcell\s+([A-Z]+\d+)\b', query_lower) is not None
            
            if is_cell_update:
                cell_guidance = {
                    "role": "system",
                    "content": "IMPORTANT: This is a request to update a specific cell. You MUST use the update_sheet action with parameters cell, value, and sheet_name."
                }
                messages.append(cell_guidance)
            else:
                # Extract the value name being updated
                value_name_match = re.search(r'update\s+(\w+)', query_lower)
                if value_name_match:
                    value_name = value_name_match.group(1)
                    named_value_guidance = {
                        "role": "system",
                        "content": f"IMPORTANT: This is a request to update a named value '{value_name}'. You MUST use the update_named_value action with parameters value_name, value, and sheet_name."
                    }
                    messages.append(named_value_guidance)
        
        # Make the API call
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using a current model with context memory
            messages=messages,
            temperature=0.7,  # More creative for conversation
            max_tokens=8000
        )
        
        # Extract the response content
        content = response.choices[0].message['content'].strip()
        
        # Parse the response for any actions
        action_info = parse_action_from_response(content)
        
        # For update requests, make sure we're using the correct action
        if is_update_request and action_info:
            # Check if we got a cell update request but LLM used wrong action
            cell_match = re.search(r'\bcell\s+([A-Z]+\d+)\b', query_lower)
            if cell_match and action_info.get("action", "").lower() != "update_sheet":
                # Extract parameters from the query directly
                cell = cell_match.group(1)
                value_match = re.search(r'to\s+(\d+[\d,.]*)', query_lower)
                value = value_match.group(1) if value_match else None
                sheet_match = re.search(r'in\s+(\w+(?:\s+\w+)*)\s+sheet', query_lower)
                sheet_name = sheet_match.group(1) if sheet_match else None
                
                if cell and value and sheet_name:
                    # Override with correct action
                    action_info = {
                        "action": "update_sheet",
                        "parameters": {
                            "cell": cell,
                            "value": value,
                            "sheet_name": sheet_name
                        },
                        "explanation": f"I'll update cell {cell} to {value} in {sheet_name} sheet."
                    }
                    logger.info(f"Overrode action to update_sheet for cell update request: {query}")
            
            # Check if we got a named value update but LLM used wrong action 
            elif not cell_match and action_info.get("action", "").lower() != "update_named_value":
                # Try to extract parameters
                value_name_match = re.search(r'update\s+(\w+)', query_lower)
                value_match = re.search(r'to\s+(\d+[\d,.]*)', query_lower)
                sheet_match = re.search(r'in\s+(\w+(?:\s+\w+)*)\s+sheet', query_lower)
                
                value_name = value_name_match.group(1) if value_name_match else None
                value = value_match.group(1) if value_match else None
                sheet_name = sheet_match.group(1) if sheet_match else None
                
                if value_name and value:
                    # Override with correct action
                    action_info = {
                        "action": "update_named_value",
                        "parameters": {
                            "value_name": value_name,
                            "value": value,
                            "sheet_name": sheet_name or "Income Statement"  # Default if not specified
                        },
                        "explanation": f"I'll update {value_name} to {value} in {sheet_name or 'Income Statement'} sheet."
                    }
                    logger.info(f"Overrode action to update_named_value for named value update request: {query}")
        
        # Update the response content if we changed the action
        if action_info:
            # Add the modified action back to the content
            action_block = f"ACTION: {action_info['action']}\nPARAMETERS: {json.dumps(action_info['parameters'])}\nEXPLANATION: {action_info['explanation']}"
            
            # Replace any existing action block or add a new one
            if "ACTION:" in content:
                content = re.sub(r'ACTION:.*?(?=ACTION:|$)', action_block, content, flags=re.DOTALL)
            else:
                content += f"\n\n{action_block}"
        
        # Add the assistant's response to the history
        session.add_message("assistant", content)
        
        # Save debugging information
        debug_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "response": content,
            "action": action_info,
            "session_id": session.session_id,
            "context": session.context,
            "system_prompt": session._get_system_prompt(),
            "financial_data_summary": {
                "tables_count": len(session.financial_data.get('tables', [])) if session.financial_data else 0,
                "key_value_pairs_count": len(session.financial_data.get('key_value_pairs', [])) if session.financial_data else 0
            },
            "filled_matrix_shape": f"{len(session.filled_matrix)}x{len(session.filled_matrix[0]) if session.filled_matrix and len(session.filled_matrix) > 0 else 0}" if session.filled_matrix else "None"
        }
        
        debug_file = f"query_debug_{int(time.time())}.json"
        try:
            with open(debug_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
            logger.info(f"Query debug data saved to {debug_file}")
        except Exception as debug_err:
            logger.warning(f"Could not save query debug data: {debug_err}")
        
        return {
            "response": content,
            "action": action_info,
            "debug_file": debug_file
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

def execute_action(session, action_info, spreadsheet_id=None):
    """
    Execute the requested action
    
    Args:
        session (QuerySession): Current query session
        action_info (dict): Action details extracted from response
        spreadsheet_id (str, optional): Google Sheet ID
        
    Returns:
        dict: Result of the action execution
    """
    if not action_info:
        return {"success": False, "message": "No action to execute"}
    
    action = action_info.get("action", "").lower()
    params = action_info.get("parameters", {})
    
    try:
        # Import sheets_manager functionality only when needed
        from sheets_manager import extract_spreadsheet_id, get_credentials
        from pdf_processor import get_sheet_data, get_sheet_names, update_sheet_data
        from googleapiclient.discovery import build
        
        # Get spreadsheet ID if not provided
        if not spreadsheet_id and session.sheet_url:
            spreadsheet_id = extract_spreadsheet_id(session.sheet_url)
            logger.info(f"Using spreadsheet ID: {spreadsheet_id} from URL: {session.sheet_url}")
        
        if not spreadsheet_id:
            return {"success": False, "message": "No Google Sheet connected"}
        
        # Initialize the sheets API service
        try:
            # Use our custom get_credentials that looks for credentials.json
            creds = get_credentials()
            service = build('sheets', 'v4', credentials=creds)
            logger.info("Successfully built sheets service")
        except Exception as e:
            logger.error(f"Error building sheets service: {str(e)}")
            return {"success": False, "message": f"Error connecting to Google Sheets API: {str(e)}"}
            
        # Helper function to find the best matching sheet name
        def find_best_matching_sheet(requested_name):
            """Find the best matching sheet name from available sheets"""
            logger.info(f"Finding match for sheet name: '{requested_name}'")
            
            # Get all sheet names if not already available
            all_sheets = []
            try:
                # Get directly from API first (most reliable)
                spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
                all_sheets = [sheet.get('properties', {}).get('title', '') 
                             for sheet in spreadsheet.get('sheets', [])]
                logger.info(f"Got sheets directly from API: {all_sheets}")
                
                # Update session available sheets
                session.available_sheets = all_sheets
            except Exception as api_error:
                logger.error(f"Error getting sheets from API: {str(api_error)}")
                
                # Try from session
                all_sheets = session.available_sheets or []
                logger.info(f"Using sheets from session: {all_sheets}")
                
                # If still empty, try get_sheet_names
                if not all_sheets:
                    try:
                        all_sheets = get_sheet_names(spreadsheet_id) or []
                        logger.info(f"Got sheets using get_sheet_names: {all_sheets}")
                        session.available_sheets = all_sheets
                    except Exception as e:
                        logger.error(f"Error getting sheet names: {str(e)}")
            
            if not all_sheets:
                logger.warning("No sheets found!")
                return None
                
            logger.info(f"Available sheets: {all_sheets}")
            
            # If exact match exists, use it
            if requested_name in all_sheets:
                logger.info(f"Found exact match: {requested_name}")
                return requested_name
                
            # Normalize the requested name (lowercase, remove spaces)
            normalized_request = requested_name.lower().replace(" ", "")
            logger.info(f"Normalized request: {normalized_request}")
            
            # Check for close matches
            for sheet in all_sheets:
                normalized_sheet = sheet.lower().replace(" ", "")
                logger.info(f"Comparing with: {sheet} (normalized: {normalized_sheet})")
                
                # Exact match after normalization
                if normalized_sheet == normalized_request:
                    logger.info(f"Found normalized match: {sheet}")
                    return sheet
                    
                # Partial match (sheet name contains request or vice versa)
                if normalized_request in normalized_sheet or normalized_sheet in normalized_request:
                    logger.info(f"Found partial match: {sheet}")
                    return sheet
                    
                # Common variations
                if ("income" in normalized_request and "statement" in normalized_sheet) or \
                   ("income" in normalized_sheet and "statement" in normalized_request) or \
                   ("balance" in normalized_request and "sheet" in normalized_sheet) or \
                   ("balance" in normalized_sheet and "sheet" in normalized_request) or \
                   ("cash" in normalized_request and "flow" in normalized_sheet) or \
                   ("cash" in normalized_sheet and "flow" in normalized_request):
                    logger.info(f"Found common variation match: {sheet}")
                    return sheet
            
            # If no match found, return the first sheet (default)
            if all_sheets:
                logger.info(f"No match found, using first sheet: {all_sheets[0]}")
                return all_sheets[0]
            
            logger.warning("No sheets available to match with")
            return None
        
        # Handle migrate_matrix action for migrating the entire matrix to a sheet
        if action == "migrate_matrix":
            target_sheet_name = params.get("sheet_name", "Financial Data")
            create_new = params.get("create_new", True)
            source_sheet = params.get("source_sheet")  # Optional parameter to specify which matrix to use
            
            # Determine which matrix to use
            matrix_to_migrate = None
            
            # First check if we have session data with mapping results
            try:
                from flask import session as flask_session
                has_session = 'all_mapping_results' in flask_session
            except Exception as e:
                logger.warning(f"Could not access Flask session: {str(e)}")
                has_session = False
                
            if has_session:
                try:
                    # Get the mapping results from the session
                    all_mapping_results = json.loads(flask_session['all_mapping_results'])
                    logger.info(f"Found {len(all_mapping_results)} mapping results in session")
                    
                    # Check if our target sheet is in the mapping results
                    for sheet_result in all_mapping_results:
                        sheet_name = sheet_result.get('sheet_name')
                        
                        # Check for match (exact or case-insensitive)
                        if (source_sheet and (sheet_name == source_sheet or 
                                             sheet_name.lower() == source_sheet.lower())) or \
                           (not source_sheet and (sheet_name == target_sheet_name or 
                                                 sheet_name.lower() == target_sheet_name.lower())):
                            
                            # Only use if status is success
                            if sheet_result.get('status') == 'success':
                                matrix_to_migrate = sheet_result.get('filled_matrix', [])
                                if matrix_to_migrate:
                                    logger.info(f"Found matrix for '{sheet_name}' in session mapping results")
                                    if source_sheet != sheet_name:
                                        source_sheet = sheet_name  # Use the exact sheet name
                                    break
                except Exception as e:
                    logger.error(f"Error accessing session mapping results: {str(e)}")
            
            # If we didn't find a matrix in the session, fall back to regular approach
            if not matrix_to_migrate:
                logger.info("No matrix found in session, using regular approach")
                
                # Simple approach: If source_sheet is specified, try to find that exact matrix
                if source_sheet:
                    logger.info(f"Looking for matrix for source sheet: {source_sheet}")
                    
                    # Check if we have this matrix directly
                    if source_sheet in session.sheet_matrices:
                        matrix_to_migrate = session.sheet_matrices[source_sheet]
                        logger.info(f"Found exact matrix for '{source_sheet}'")
                    else:
                        # Try case-insensitive matching
                        for name, matrix in session.sheet_matrices.items():
                            if name.lower() == source_sheet.lower():
                                matrix_to_migrate = matrix
                                source_sheet = name  # Use the actual name with correct case
                                logger.info(f"Found case-insensitive match: '{name}'")
                                break
                                
                        if not matrix_to_migrate:
                            # As a fallback, use get_matrix_for_sheet
                            matrix_to_migrate = session.get_matrix_for_sheet(source_sheet)
                            if matrix_to_migrate:
                                logger.info(f"Found matrix using get_matrix_for_sheet")
                else:
                    # If no source specified, use target sheet name if it matches a matrix
                    if target_sheet_name in session.sheet_matrices:
                        matrix_to_migrate = session.sheet_matrices[target_sheet_name]
                        source_sheet = target_sheet_name
                        logger.info(f"Using matrix matching target sheet name: '{target_sheet_name}'")
                    # Otherwise use the default filled matrix
                    else:
                        matrix_to_migrate = session.filled_matrix
                        logger.info("No source sheet specified, using default filled matrix")
                    
                # Final check - if we still don't have a matrix, use the first available one
                if matrix_to_migrate is None and session.sheet_matrices:
                    first_key = next(iter(session.sheet_matrices.keys()))
                    matrix_to_migrate = session.sheet_matrices[first_key]
                    source_sheet = first_key
                    logger.info(f"Using first available matrix: '{first_key}'")
            
            # If we still don't have a matrix, return an error
            if matrix_to_migrate is None:
                available_keys = list(session.sheet_matrices.keys())
                return {"success": False, "message": f"No matrix available to migrate. Available matrices: {available_keys}"}
                
            # Log what we're using
            if matrix_to_migrate:
                matrix_shape = f"{len(matrix_to_migrate)}x{len(matrix_to_migrate[0]) if matrix_to_migrate and len(matrix_to_migrate) > 0 else 0}"
                logger.info(f"Using matrix for '{source_sheet or 'default'}': {matrix_shape}")

            logger.info(f"Migrating matrix from '{source_sheet or 'default'}' to sheet '{target_sheet_name}' (create_new={create_new})")
            
            try:
                # Get service (already initialized above)
                
                if create_new:
                    # Import create_new_sheet
                    from sheets_manager import create_new_sheet
                    
                    # Generate a unique timestamp for the sheet name if not provided
                    if target_sheet_name == "Financial Data":
                        timestamp = uuid.uuid4().hex[:8]
                        target_sheet_name = f"Financial_Data_{timestamp}"
                    
                    logger.info(f"Creating new sheet '{target_sheet_name}' with matrix data")
                    
                    # Create a new sheet with the specified matrix
                    result = create_new_sheet(
                        service,
                        spreadsheet_id,
                        target_sheet_name,
                        headers=None,  # No separate headers, everything is in the matrix
                        data=matrix_to_migrate
                    )
                    
                    return {
                        "success": True,
                        "message": f"Successfully migrated {source_sheet or 'default'} matrix to new sheet '{target_sheet_name}'",
                        "sheet_name": target_sheet_name,
                        "source_sheet": source_sheet,
                        "rows_migrated": len(matrix_to_migrate),
                        "columns_migrated": len(matrix_to_migrate[0]) if matrix_to_migrate and len(matrix_to_migrate) > 0 else 0
                    }
                else:
                    # Use EXACT same code as in app.py map-to-sheet route
                    try:
                        # First, clear the entire sheet
                        logger.info(f"Clearing sheet '{target_sheet_name}'")
                        try:
                            clear_request = service.spreadsheets().values().clear(
                                spreadsheetId=spreadsheet_id,
                                range=f"{target_sheet_name}"
                            )
                            clear_response = clear_request.execute()
                            logger.info(f"Clear response: {clear_response}")
                        except Exception as clear_e:
                            logger.error(f"Error clearing sheet: {str(clear_e)}")
                            # Continue anyway - sheet might not exist yet
                        
                        # Then update with the new matrix
                        logger.info(f"Updating sheet '{target_sheet_name}' with matrix from '{source_sheet or 'default'}' ({len(matrix_to_migrate)}x{len(matrix_to_migrate[0]) if matrix_to_migrate and len(matrix_to_migrate) > 0 else 0})")
                        
                        update_range = f"{target_sheet_name}!A1"
                        body = {
                            'values': matrix_to_migrate
                        }
                        
                        # Print first few rows for debugging
                        sample_data = str(matrix_to_migrate[:2]) if matrix_to_migrate and len(matrix_to_migrate) >= 2 else "Empty matrix"
                        logger.info(f"Sample data: {sample_data}")
                        
                        update_result = service.spreadsheets().values().update(
                            spreadsheetId=spreadsheet_id,
                            range=update_range,
                            valueInputOption='USER_ENTERED',
                            body=body
                        ).execute()
                        
                        logger.info(f"Update result: {update_result}")
                        
                        return {
                            "success": True,
                            "message": f"Successfully replaced entire content of sheet '{target_sheet_name}' with {source_sheet or 'default'} matrix data",
                            "sheet_name": target_sheet_name,
                            "source_sheet": source_sheet,
                            "rows_migrated": len(matrix_to_migrate),
                            "columns_migrated": len(matrix_to_migrate[0]) if matrix_to_migrate and len(matrix_to_migrate) > 0 else 0,
                            "cells_updated": update_result.get('updatedCells', 0),
                            "update_result": update_result
                        }
                    except Exception as api_e:
                        logger.error(f"Error with direct API calls: {str(api_e)}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        
                        return {
                            "success": False,
                            "message": f"Error migrating matrix: {str(api_e)}",
                            "error_details": traceback.format_exc()
                        }
            
            except Exception as e:
                logger.error(f"Error migrating matrix to sheet: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {
                    "success": False, 
                    "message": f"Error migrating matrix to sheet: {str(e)}",
                    "error_details": traceback.format_exc()
                }
        
        # Handle find_cell action for finding the right cell based on text search
        elif action == "find_cell":
            search_term = params.get("search_term", "").lower()
            requested_sheet_name = params.get("sheet_name", session.sheet_name or "Sheet1")
            
            # Find the best matching sheet name
            sheet_name = find_best_matching_sheet(requested_sheet_name)
            
            if not sheet_name:
                return {"success": False, "message": f"No sheets found in the spreadsheet"}
            
            if not search_term:
                return {"success": False, "message": "Missing search term for find_cell action"}
            
            # Get the sheet data
            sheet_data = get_sheet_data(spreadsheet_id, sheet_name)
            
            if not sheet_data:
                return {"success": False, "message": f"Sheet '{sheet_name}' not found or is empty"}
            
            # Search for the term in the first column (typically contains row labels)
            matching_rows = []
            for row_idx, row in enumerate(sheet_data):
                if not row:
                    continue
                
                # Check first cell (usually labels) for the search term
                first_cell = str(row[0]).lower() if row and len(row) > 0 else ""
                if search_term in first_cell:
                    row_preview = " | ".join([str(cell) for cell in row[:5]])
                    if len(row) > 5:
                        row_preview += " | ..."
                    
                    matching_rows.append({
                        "row_index": row_idx,
                        "row_label": row[0],
                        "row_preview": row_preview,
                        "row_data": row,
                        "cell_ref": f"A{row_idx + 1}"  # Convert to 1-based cell reference
                    })
            
            # Check if we found any matches
            if not matching_rows:
                return {
                    "success": False, 
                    "message": f"No rows containing '{search_term}' found in sheet '{sheet_name}'"
                }
            
            # Return the matching rows
            return {
                "success": True,
                "message": f"Found {len(matching_rows)} matching rows for '{search_term}' in sheet '{sheet_name}'",
                "sheet_name": sheet_name,
                "matching_rows": matching_rows
            }
            
        # Handle update_sheet action
        elif action == "update_sheet":
            requested_sheet_name = params.get("sheet_name", session.sheet_name or "Sheet1")
            cell = params.get("cell")
            value = params.get("value")
            
            # Debug logging
            logger.info(f"Updating cell '{cell}' to '{value}' in sheet: {requested_sheet_name}")
            logger.info(f"Available sheets in session: {session.available_sheets}")
            
            if not cell or value is None:
                return {"success": False, "message": "Missing required parameters for update_sheet"}
            
            # Find the best matching sheet name
            sheet_name = find_best_matching_sheet(requested_sheet_name)
            
            if not sheet_name:
                return {"success": False, "message": f"No sheets found in the spreadsheet"}
            
            # Create a direct service connection using credentials.json
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                import os  # Ensure we have os imported
                
                # Use credentials.json directly instead of service_account.json
                credentials_path = 'credentials.json'
                logger.info(f"Using credentials from: {credentials_path}")
                
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        credentials_path,
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    logger.info(f"Successfully loaded credentials from {credentials_path}")
                except Exception as cred_error:
                    logger.error(f"Error loading credentials from {credentials_path}: {str(cred_error)}")
                    # Try with absolute path
                    abs_credentials_path = os.path.abspath(credentials_path)
                    logger.info(f"Trying absolute path: {abs_credentials_path}")
                    credentials = service_account.Credentials.from_service_account_file(
                        abs_credentials_path,
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    logger.info(f"Successfully loaded credentials from absolute path: {abs_credentials_path}")
                
                # Build the service using our direct credentials
                direct_service = build('sheets', 'v4', credentials=credentials)
                service = direct_service
                logger.info("Successfully created direct_service for update_sheet action")
                
                # Make a direct cell update
                # Get the range for the update
                safe_sheet_name = sheet_name.replace("'", "''")
                cell_range = f"'{safe_sheet_name}'!{cell}"
                logger.info(f"Updating range: {cell_range} with value: {value}")
                
                # Perform the update
                body = {
                    'values': [[value]]  # Single cell update
                }
                
                update_result = service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=cell_range,
                    valueInputOption='USER_ENTERED',
                    body=body
                ).execute()
                
                # Also update the matrix in session if we have it
                if session.sheet_matrices and sheet_name in session.sheet_matrices:
                    matrix = session.sheet_matrices[sheet_name]
                    
                    # Convert cell reference (e.g., B5) to row and column indices
                    # Extract column letter and row number
                    col_letter = ''.join(c for c in cell if c.isalpha()).upper()
                    row_num = int(''.join(c for c in cell if c.isdigit())) - 1  # Convert to 0-based
                    
                    # Convert column letter to number (A=0, B=1, etc.)
                    col_num = 0
                    for i, char in enumerate(reversed(col_letter)):
                        col_num += (ord(char) - ord('A') + 1) * (26 ** i)
                    col_num -= 1  # Convert to 0-based
                    
                    logger.info(f"Converted cell {cell} to row={row_num}, col={col_num}")
                    
                    if 0 <= row_num < len(matrix):
                        # Ensure the matrix row has enough columns
                        while len(matrix[row_num]) <= col_num:
                            matrix[row_num].append("")
                        # Update the value
                        matrix[row_num][col_num] = value
                        logger.info(f"Updated matrix in memory at row {row_num}, col {col_num}")
                
                # Also update filled_matrix if it matches
                if session.filled_matrix:
                    if 0 <= row_num < len(session.filled_matrix) and 0 <= col_num:
                        if len(session.filled_matrix[row_num]) > col_num:
                            session.filled_matrix[row_num][col_num] = value
                            logger.info(f"Updated filled_matrix in memory at row {row_num}, col {col_num}")
                
                return {
                    "success": True,
                    "message": f"Successfully updated cell {cell} in sheet '{sheet_name}' to value '{value}'",
                    "updated_cells": update_result.get('updatedCells', 1),
                    "sheet_name": sheet_name
                }
                
            except Exception as e:
                logger.error(f"Error updating cell: {str(e)}")
                logger.error(traceback.format_exc())
                return {"success": False, "message": f"Error updating cell: {str(e)}"}
        
        # Handle debug_sheets action
        elif action == "debug_sheets":
            """Special action to debug sheet access issues"""
            try:
                # Get credentials and build service
                creds = get_credentials()
                service = build('sheets', 'v4', credentials=creds)
                
                # Get all sheet names
                sheet_names = get_sheet_names(spreadsheet_id) or []
                
                # Test access to each sheet
                sheet_access_results = []
                for sheet_name in sheet_names:
                    try:
                        sheet_data = get_sheet_data(spreadsheet_id, sheet_name)
                        rows = len(sheet_data) if sheet_data else 0
                        cols = len(sheet_data[0]) if sheet_data and len(sheet_data) > 0 else 0
                        sheet_access_results.append({
                            "sheet_name": sheet_name,
                            "accessible": True,
                            "rows": rows,
                            "columns": cols,
                            "sample": sheet_data[:2] if sheet_data and len(sheet_data) > 0 else []
                        })
                    except Exception as e:
                        sheet_access_results.append({
                            "sheet_name": sheet_name,
                            "accessible": False,
                            "error": str(e)
                        })
                
                # Update session available sheets
                session.available_sheets = sheet_names
                
                # Get spreadsheet metadata
                try:
                    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
                    spreadsheet_title = spreadsheet.get('properties', {}).get('title', 'Unknown')
                except Exception as e:
                    spreadsheet_title = "Error getting title: " + str(e)
                
                return {
                    "success": True,
                    "message": f"Found {len(sheet_names)} sheets in spreadsheet",
                    "spreadsheet_id": spreadsheet_id,
                    "spreadsheet_title": spreadsheet_title,
                    "sheet_names": sheet_names,
                    "sheet_access_results": sheet_access_results
                }
            except Exception as e:
                logger.error(f"Error debugging sheets: {str(e)}")
                return {"success": False, "message": f"Error debugging sheets: {str(e)}"}
        
        # Handle migrate_all_matrices action
        elif action == "migrate_all_matrices":
            """Migrate all available matrices to their corresponding sheets"""
            try:
                # Check if we have session data with mapping results
                # Import Flask session
                try:
                    from flask import session as flask_session
                    has_session = 'all_mapping_results' in flask_session
                except Exception as e:
                    logger.warning(f"Could not access Flask session: {str(e)}")
                    has_session = False
                
                if has_session:
                    # Use the exact same approach as the UI's map-to-sheet endpoint
                    try:
                        all_mapping_results = json.loads(flask_session['all_mapping_results'])
                        logger.info(f"Found {len(all_mapping_results)} mapping results in session")
                        
                        # Initialize service if not already
                        creds = get_credentials()
                        service = build('sheets', 'v4', credentials=creds)
                        
                        # Track results for each migration
                        migration_results = []
                        
                        # Process each sheet from the mapping results
                        for sheet_result in all_mapping_results:
                            sheet_name = sheet_result.get('sheet_name')
                            
                            # Skip sheets that had errors
                            if sheet_result.get('status') != 'success':
                                logger.warning(f"Skipping sheet '{sheet_name}' due to status: {sheet_result.get('status')}")
                                migration_results.append({
                                    'sheet_name': sheet_name,
                                    'status': 'skipped',
                                    'message': f"Sheet was skipped due to previous errors: {sheet_result.get('error', 'Unknown error')}"
                                })
                                continue
                            
                            filled_matrix = sheet_result.get('filled_matrix', [])
                            stats = sheet_result.get('stats', {})
                            
                            if not filled_matrix:
                                logger.warning(f"Skipping sheet '{sheet_name}' - empty matrix")
                                migration_results.append({
                                    'sheet_name': sheet_name,
                                    'status': 'skipped',
                                    'message': "Empty matrix"
                                })
                                continue
                            
                            logger.info(f"Migrating matrix for sheet '{sheet_name}' ({len(filled_matrix)}x{len(filled_matrix[0]) if filled_matrix and len(filled_matrix) > 0 else 0})")
                            
                            try:
                                # First, clear the entire sheet
                                logger.info(f"Clearing sheet '{sheet_name}'")
                                try:
                                    clear_request = service.spreadsheets().values().clear(
                                        spreadsheetId=spreadsheet_id,
                                        range=f"{sheet_name}"
                                    )
                                    clear_response = clear_request.execute()
                                    logger.info(f"Clear response: {clear_response}")
                                except Exception as clear_e:
                                    logger.warning(f"Error clearing sheet '{sheet_name}': {str(clear_e)}")
                                
                                # Then update with the new matrix
                                update_range = f"{sheet_name}!A1"
                                body = {
                                    'values': filled_matrix
                                }
                                
                                update_result = service.spreadsheets().values().update(
                                    spreadsheetId=spreadsheet_id,
                                    range=update_range,
                                    valueInputOption='USER_ENTERED',
                                    body=body
                                ).execute()
                                
                                logger.info(f"Update result for '{sheet_name}': {update_result}")
                                
                                migration_results.append({
                                    'sheet_name': sheet_name,
                                    'status': 'replaced',
                                    'stats': stats,
                                    'message': f'Replaced sheet: {sheet_name}'
                                })
                            except Exception as e:
                                logger.error(f"Error migrating matrix to sheet '{sheet_name}': {str(e)}")
                                migration_results.append({
                                    'sheet_name': sheet_name,
                                    'status': 'error',
                                    'message': f'Error updating sheet {sheet_name}: {str(e)}'
                                })
                        
                        # Calculate success summary
                        successful_migrations = sum(1 for result in migration_results if result.get('status') == 'replaced')
                        
                        return {
                            "success": successful_migrations > 0,
                            "message": f"Successfully migrated {successful_migrations} out of {len(migration_results)} sheets",
                            "migration_results": migration_results
                        }
                    except Exception as e:
                        logger.error(f"Error processing session mapping results: {str(e)}")
                        # Fall back to the regular approach
                        logger.info("Falling back to regular migration approach")
                
                # First verify we have matrices and sheets
                if not session.sheet_matrices:
                    return {"success": False, "message": "No matrices available to migrate"}
                
                if not session.available_sheets:
                    # Try to get available sheets
                    try:
                        session.available_sheets = get_sheet_names(spreadsheet_id) or []
                    except Exception as e:
                        return {"success": False, "message": f"Could not get available sheets: {str(e)}"}
                
                # Initialize service if not already
                creds = get_credentials()
                service = build('sheets', 'v4', credentials=creds)
                
                # Track results for each migration
                migration_results = []
                
                # Simple approach: Just migrate what we have from the mapping
                matrices_to_migrate = []
                
                # Check if we have any preview matrices (from PDF mapping)
                preview_matrices = {}
                for name, matrix in session.sheet_matrices.items():
                    if name.endswith("_preview"):
                        actual_name = name.replace("_preview", "")
                        preview_matrices[actual_name] = matrix
                        logger.info(f"Found preview matrix for '{actual_name}'")
                
                # If we have preview matrices, use them
                if preview_matrices:
                    matrices_to_migrate = list(preview_matrices.items())
                    logger.info(f"Using {len(matrices_to_migrate)} preview matrices for migration")
                else:
                    # Otherwise just use the existing matrices
                    # Remove any "preview" or "default" entries
                    for name, matrix in session.sheet_matrices.items():
                        if not name.endswith("_preview") and name != "default":
                            matrices_to_migrate.append((name, matrix))
                    
                    logger.info(f"Using {len(matrices_to_migrate)} existing matrices for migration")
                
                # If still no matrices, try using the filled_matrix for each sheet
                if not matrices_to_migrate and session.filled_matrix:
                    logger.info("No matrices to migrate, using filled_matrix for all sheets")
                    for sheet_name in session.available_sheets:
                        matrices_to_migrate.append((sheet_name, session.filled_matrix))
                
                # Perform the migrations
                for sheet_name, matrix in matrices_to_migrate:
                    logger.info(f"Migrating matrix to sheet '{sheet_name}'")
                    
                    # Check if this sheet exists
                    if sheet_name not in session.available_sheets:
                        logger.warning(f"Sheet '{sheet_name}' not found in available sheets")
                        # Try to find a case-insensitive match
                        matched = False
                        for available_sheet in session.available_sheets:
                            if sheet_name.lower() == available_sheet.lower():
                                sheet_name = available_sheet
                                matched = True
                                logger.info(f"Using case-insensitive match: '{available_sheet}'")
                                break
                        
                        if not matched:
                            logger.warning(f"No matching sheet found for '{sheet_name}', skipping")
                            migration_results.append({
                                "sheet_name": sheet_name,
                                "success": False,
                                "error": "Sheet not found"
                            })
                            continue
                    
                    try:
                        # Clear the sheet first
                        try:
                            clear_request = service.spreadsheets().values().clear(
                                spreadsheetId=spreadsheet_id,
                                range=f"{sheet_name}"
                            )
                            clear_response = clear_request.execute()
                            logger.info(f"Cleared sheet '{sheet_name}': {clear_response}")
                        except Exception as clear_e:
                            logger.warning(f"Error clearing sheet '{sheet_name}': {str(clear_e)}")
                        
                        # Update with the matrix data
                        update_range = f"{sheet_name}!A1"
                        body = {
                            'values': matrix
                        }
                        
                        update_result = service.spreadsheets().values().update(
                            spreadsheetId=spreadsheet_id,
                            range=update_range,
                            valueInputOption='USER_ENTERED',
                            body=body
                        ).execute()
                        
                        logger.info(f"Updated sheet '{sheet_name}': {update_result}")
                        
                        migration_results.append({
                            "sheet_name": sheet_name,
                            "success": True,
                            "rows": len(matrix),
                            "columns": len(matrix[0]) if matrix and len(matrix) > 0 else 0,
                            "cells_updated": update_result.get('updatedCells', 0)
                        })
                    except Exception as e:
                        logger.error(f"Error migrating matrix to sheet '{sheet_name}': {str(e)}")
                        migration_results.append({
                            "sheet_name": sheet_name,
                            "success": False,
                            "error": str(e)
                        })
                
                # Calculate success summary
                successful_migrations = sum(1 for result in migration_results if result.get("success", False))
                
                return {
                    "success": successful_migrations > 0,
                    "message": f"Successfully migrated {successful_migrations} out of {len(migration_results)} matrices",
                    "migration_results": migration_results
                }
            except Exception as e:
                logger.error(f"Error in migrate_all_matrices: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {
                    "success": False,
                    "message": f"Error migrating all matrices: {str(e)}",
                    "error_details": traceback.format_exc()
                }
        
        # Handle list_sheets action
        elif action == "list_sheets":
            # Get all available sheets in the spreadsheet
            sheet_names = get_sheet_names(spreadsheet_id)
            
            if not sheet_names:
                return {"success": False, "message": "No sheets found in the spreadsheet"}
            
            # Update the session's available sheets
            session.available_sheets = sheet_names
            
            # Get the active sheet name
            active_sheet = session.sheet_name or sheet_names[0]
            
            return {
                "success": True,
                "message": f"Found {len(sheet_names)} sheets in the spreadsheet",
                "sheets": sheet_names,
                "active_sheet": active_sheet
            }
        
        # Handle update_named_value action for updating a value by name instead of cell reference
        elif action == "update_named_value":
            value_name = params.get("value_name")
            new_value = params.get("value")
            requested_sheet_name = params.get("sheet_name", session.sheet_name or "Sheet1")
            
            # Debug logging
            logger.info(f"Updating named value '{value_name}' to '{new_value}' in sheet: {requested_sheet_name}")
            logger.info(f"Available sheets in session: {session.available_sheets}")
            
            if not value_name or new_value is None:
                return {"success": False, "message": "Missing required parameters: value_name and value"}
            
            # Find the best matching sheet
            sheet_name = find_best_matching_sheet(requested_sheet_name)
            if not sheet_name:
                available_sheets = get_sheet_names(spreadsheet_id) or []
                sheet_list = ", ".join(available_sheets) if available_sheets else "No sheets found"
                return {"success": False, "message": f"Could not find a sheet matching '{requested_sheet_name}'. Available sheets: {sheet_list}"}
                
            logger.info(f"Found matching sheet: '{sheet_name}'")
            
            try:
                # Create a direct service connection using credentials.json
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                import os  # Add missing import
                
                # Use credentials.json directly instead of service_account.json
                credentials_path = 'credentials.json'
                logger.info(f"Using credentials from: {credentials_path}")
                
                try:
                    credentials = service_account.Credentials.from_service_account_file(
                        credentials_path,
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    logger.info(f"Successfully loaded credentials from {credentials_path}")
                except Exception as cred_error:
                    logger.error(f"Error loading credentials from {credentials_path}: {str(cred_error)}")
                    # Try with absolute path
                    import os  # Redundant but ensuring it's available
                    abs_credentials_path = os.path.abspath(credentials_path)
                    logger.info(f"Trying absolute path: {abs_credentials_path}")
                    credentials = service_account.Credentials.from_service_account_file(
                        abs_path,
                        scopes=['https://www.googleapis.com/auth/spreadsheets']
                    )
                    logger.info(f"Successfully loaded credentials from absolute path: {abs_credentials_path}")
                
                # Build the service using our direct credentials
                direct_service = build('sheets', 'v4', credentials=credentials)
                service = direct_service
                
                # Step 1: Get the current sheet data/structure
                logger.info(f"Fetching current data for sheet: '{sheet_name}'")
                
                # Get the full sheet data
                sheet_range = f"'{sheet_name}'"
                result = service.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=sheet_range
                ).execute()
                
                current_matrix = result.get('values', [])
                
                if not current_matrix:
                    return {"success": False, "message": f"Sheet '{sheet_name}' is empty or has no data"}
                
                # Log the current matrix size
                matrix_size = f"{len(current_matrix)}x{len(current_matrix[0]) if current_matrix and len(current_matrix) > 0 else 0}"
                logger.info(f"Retrieved current matrix: {matrix_size}")
                
                # Step 2: Send to OpenAI for updating
                # Ensure os module is imported
                import os
                import openai
                
                openai.api_key = os.getenv("OPENAI_API_KEY")
                if not openai.api_key:
                    return {"success": False, "message": "OpenAI API key is required but not found in environment"}
                
                # Convert matrix to string representation for the prompt
                matrix_str = json.dumps(current_matrix)
                
                # Create the prompt
                prompt = f"""
You are given a spreadsheet matrix from a financial sheet named '{sheet_name}'.
The user wants to update the value named '{value_name}' to '{new_value}'.

Current spreadsheet matrix:
{matrix_str}

Please update the matrix to reflect this change. Return your response in the following JSON format:
{{
    "updated_matrix": [the complete updated matrix with the change],
    "cells_changed": [number of cells that were changed]
}}

Make sure to return valid JSON that can be parsed. The updated_matrix should include ALL rows and columns from the original.
"""
                
                logger.info("Sending matrix to OpenAI for update")
                
                # Call OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a financial data expert that updates spreadsheets."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.1
                )
                
                # Extract the response content
                ai_response = response.choices[0].message['content'].strip()
                logger.info(f"Received response from OpenAI: {ai_response[:100]}...")
                
                # Step 3: Parse the response
                # Find and extract the JSON part from the response
                json_match = re.search(r'({.*})', ai_response, re.DOTALL)
                
                if not json_match:
                    logger.error(f"Could not find JSON in OpenAI response: {ai_response}")
                    return {"success": False, "message": "Could not parse OpenAI response"}
                
                json_str = json_match.group(1)
                
                try:
                    update_data = json.loads(json_str)
                    
                    updated_matrix = update_data.get('updated_matrix', [])
                    cells_changed = update_data.get('cells_changed', 0)
                    
                    if not updated_matrix:
                        return {"success": False, "message": "Updated matrix is empty in OpenAI response"}
                    
                    # Step 4: Update the sheet with the complete matrix
                    logger.info(f"Updating sheet '{sheet_name}' with new matrix ({len(updated_matrix)}x{len(updated_matrix[0]) if updated_matrix and len(updated_matrix) > 0 else 0})")
                    
                    # First clear the sheet
                    clear_request = service.spreadsheets().values().clear(
                        spreadsheetId=spreadsheet_id,
                        range=sheet_range
                    )
                    clear_response = clear_request.execute()
                    logger.info(f"Cleared sheet '{sheet_name}': {clear_response}")
                    
                    # Then update with the new matrix
                    body = {
                        'values': updated_matrix
                    }
                    
                    update_result = service.spreadsheets().values().update(
                        spreadsheetId=spreadsheet_id,
                        range=f"{sheet_name}!A1",
                        valueInputOption='USER_ENTERED',
                        body=body
                    ).execute()
                    
                    # Also update the matrix in session if we have it
                    if session.sheet_matrices and sheet_name in session.sheet_matrices:
                        session.sheet_matrices[sheet_name] = updated_matrix
                    
                    return {
                        "success": True,
                        "message": f"Successfully updated {value_name} to {new_value} in sheet '{sheet_name}'",
                        "updated_cells": cells_changed,
                        "sheet_name": sheet_name,
                        "api_cells_updated": update_result.get('updatedCells', 0)
                    }
                
                except json.JSONDecodeError as json_err:
                    logger.error(f"Error parsing JSON from OpenAI: {str(json_err)}")
                    logger.error(f"JSON string that failed to parse: {json_str}")
                    return {"success": False, "message": f"Error parsing OpenAI response: {str(json_err)}"}
                
            except Exception as e:
                logger.error(f"Error updating named value: {str(e)}")
                logger.error(traceback.format_exc())
                return {"success": False, "message": f"Error updating named value: {str(e)}"}
        
        # Handle regenerate_mapping action
        elif action == "regenerate_mapping":
            logger.info("Regenerating mapping for all sheets")
            
            try:
                # Check if we have a valid PDF data and spreadsheet ID
                if not session.financial_data:
                    return {"success": False, "message": "No financial data available for mapping"}
                
                if not spreadsheet_id:
                    return {"success": False, "message": "No Google Sheet connected"}
                
                # Get sheet structure for all tabs
                try:
                    # Try different import paths
                    try:
                        from sheets_manager import get_sheet_structure
                    except ModuleNotFoundError:
                        try:
                            from truffles.sheets_manager import get_sheet_structure
                        except ModuleNotFoundError:
                            raise ImportError("Could not import get_sheet_structure from any module")
                        
                    sheet_structure = get_sheet_structure(session.sheet_url)
                    logger.info(f"Got sheet structure: {len(sheet_structure.get('sheets', []))} sheets")
                except Exception as e:
                    logger.error(f"Error getting sheet structure: {str(e)}")
                    return {"success": False, "message": f"Error getting sheet structure: {str(e)}"}
                
                # Check if we have sheets to process
                if not sheet_structure.get('sheets'):
                    return {"success": False, "message": "No sheets found in the spreadsheet"}
                
                # Process each sheet tab
                all_mapping_results = []
                
                for sheet in sheet_structure.get('sheets', []):
                    sheet_name = sheet.get('title', 'Unnamed Sheet')
                    matrix = sheet.get('values', [])
                    
                    # Skip empty sheets
                    if not matrix:
                        all_mapping_results.append({
                            'sheet_name': sheet_name,
                            'error': 'Sheet is empty',
                            'status': 'skipped'
                        })
                        continue
                    
                    # Use the simple mapper to get the filled matrix for this sheet
                    try:
                        # Try different import paths
                        try:
                            from simple_mapper import map_financial_data
                        except ModuleNotFoundError:
                            try:
                                from truffles.simple_mapper import map_financial_data
                            except ModuleNotFoundError:
                                raise ImportError("Could not import map_financial_data from any module")
                            
                        logger.info(f"Mapping financial data for sheet: '{sheet_name}'")
                        mapping_result = map_financial_data(session.financial_data, matrix)
                        
                        # Store full matrix in memory for later use
                        filled_matrix = mapping_result.get('filled_matrix', [])
                        
                        # Store in session matrices
                        session.sheet_matrices[sheet_name] = filled_matrix
                        
                        # Add to results - IMPORTANT: Keep 'filled_matrix' key for UI compatibility
                        all_mapping_results.append({
                            'sheet_name': sheet_name,
                            'filled_matrix': filled_matrix,  # Use exact same key as expected by UI
                            'stats': mapping_result.get('stats', {}),
                            'status': 'success'
                        })
                        
                        logger.info(f"Successfully mapped data for sheet '{sheet_name}': {mapping_result.get('stats', {})}")
                    except Exception as e:
                        logger.error(f"Error mapping data for sheet '{sheet_name}': {str(e)}")
                        all_mapping_results.append({
                            'sheet_name': sheet_name,
                            'error': str(e),
                            'status': 'error'
                        })
                
                # Try to store mapping results in the required Flask session variables
                try:
                    from flask import session as flask_session
                    
                    # Create a preview version with limited data for session storage
                    preview_results = []
                    for result in all_mapping_results:
                        if result.get('status') == 'success':
                            # Create a copy without the full matrix
                            preview_result = result.copy()
                            full_matrix = preview_result.get('filled_matrix', [])
                            # Only keep first 5 rows for preview
                            preview_result['filled_matrix'] = full_matrix[:5] if full_matrix else []
                            preview_results.append(preview_result)
                        else:
                            # Error results don't have matrices, so include as-is
                            preview_results.append(result)
                    
                    # Store in the exact same session variable used by app.py
                    flask_session['all_mapping_results'] = json.dumps(all_mapping_results)
                    logger.info("Stored mapping results in Flask session (may be large)")
                    
                    # Also store a more compact version for backup
                    flask_session['all_mapping_results_preview'] = json.dumps(preview_results)
                    logger.info("Stored compact mapping results preview")
                    
                    # Add flags to help the UI know to refresh
                    flask_session['mapping_regenerated'] = True
                    flask_session['mapping_timestamp'] = int(time.time())
                    
                    # Force a session save
                    try:
                        flask_session.modified = True
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error storing mapping results in Flask session: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # If we have at least one successful mapping, use it as the default filled matrix
                for result in all_mapping_results:
                    if result.get('status') == 'success':
                        # Find the corresponding full matrix in session.sheet_matrices
                        sheet_name = result.get('sheet_name')
                        if sheet_name in session.sheet_matrices:
                            session.filled_matrix = session.sheet_matrices[sheet_name]
                            logger.info(f"Set default filled matrix from sheet '{sheet_name}'")
                            break
                
                # Count successful and failed mappings
                successful = sum(1 for r in all_mapping_results if r.get('status') == 'success')
                failed = sum(1 for r in all_mapping_results if r.get('status') == 'error')
                skipped = sum(1 for r in all_mapping_results if r.get('status') == 'skipped')
                
                # Create a special notification for the UI
                try:
                    from flask import session as flask_session
                    
                    # Save a notification for the frontend to display
                    notification = {
                        "type": "success",
                        "message": f"Mapping regenerated successfully for {successful} sheets. Ready to migrate to Google Sheets.",
                        "timestamp": int(time.time())
                    }
                    flask_session['notification'] = json.dumps(notification)
                    flask_session.modified = True
                except:
                    pass
                
                # Now build result for display
                status_message = f"Regenerated mapping for {successful} sheets successfully."
                if failed > 0:
                    status_message += f" {failed} sheets had errors."
                if skipped > 0:
                    status_message += f" {skipped} sheets were skipped."
                
                # Return the results
                return {
                    "success": successful > 0,
                    "message": status_message,
                    "mapping_results": all_mapping_results,  # Use key expected by UI
                    "successful_mappings": successful,
                    "failed_mappings": failed,
                    "skipped_mappings": skipped,
                    "ready_to_migrate": True,  # Indicate that migration is ready but not performed
                    "note": "The mapping has been regenerated and is ready for migration, but no data has been sent to Google Sheets yet."
                }
                
            except Exception as e:
                logger.error(f"Error regenerating mapping: {str(e)}")
                logger.error(traceback.format_exc())
                return {"success": False, "message": f"Error regenerating mapping: {str(e)}"}
        
        else:
            return {"success": False, "message": f"Unknown action: {action}"}
    
    except ImportError as e:
        logger.error(f"Import error in execute_action: {str(e)}")
        return {"success": False, "message": f"Google Sheets integration is not available: {str(e)}"}
    except Exception as e:
        logger.error(f"Error executing action: {str(e)}")
        return {"success": False, "message": f"Error executing action: {str(e)}"} 

# Add a custom get_credentials function in this file to override the one from sheets_manager
def get_credentials():
    """
    Get Google Sheets API credentials from the credentials.json file
    """
    try:
        from google.oauth2 import service_account
        import os
        
        # Look for credentials.json instead of service_account.json
        credentials_file = 'credentials.json'
        logger.info(f"Looking for credentials file: {credentials_file}")
        
        # Try with relative path first
        try:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_file,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            logger.info(f"Successfully loaded credentials from {credentials_file}")
            return credentials
        except FileNotFoundError as e:
            logger.warning(f"Could not find credentials at relative path: {e}")
            # Try with absolute path
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                abs_path = os.path.join(base_dir, credentials_file)
                logger.info(f"Trying with absolute path: {abs_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    abs_path,
                    scopes=['https://www.googleapis.com/auth/spreadsheets']
                )
                logger.info(f"Successfully loaded credentials from {abs_path}")
                return credentials
            except FileNotFoundError as e:
                logger.warning(f"Could not find credentials at base directory: {e}")
                # Try one directory up
                parent_dir = os.path.dirname(base_dir)
                abs_path = os.path.join(parent_dir, credentials_file)
                logger.info(f"Trying with parent directory: {abs_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    abs_path,
                    scopes=['https://www.googleapis.com/auth/spreadsheets']
                )
                logger.info(f"Successfully loaded credentials from {abs_path}")
                return credentials
    except Exception as e:
        logger.error(f"Failed to get credentials: {str(e)}")
        raise Exception(f"Failed to get credentials: {str(e)}")

# Override the pdf_processor's get_credentials method to use our version
try:
    import pdf_processor
    # Monkey patch the get_credentials method
    pdf_processor.get_credentials = get_credentials
    logger.info("Successfully patched pdf_processor.get_credentials")
except Exception as e:
    logger.warning(f"Could not patch pdf_processor.get_credentials: {e}")