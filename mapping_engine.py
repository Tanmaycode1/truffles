import re
import json
import difflib
import pandas as pd
from collections import defaultdict
import openai
import os
from dotenv import load_dotenv
from difflib import SequenceMatcher
import logging

load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def normalize_string(s):
    """Normalize a string for comparison by removing punctuation, spaces and converting to lowercase"""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r'[^\w]', '', s).lower()

def calculate_similarity(str1, str2):
    """Calculate the similarity ratio between two strings"""
    if not str1 or not str2:
        return 0
    
    # Normalize strings for comparison
    norm1 = normalize_string(str1)
    norm2 = normalize_string(str2)
    
    if not norm1 or not norm2:
        return 0
    
    # Use difflib's SequenceMatcher for string similarity
    return difflib.SequenceMatcher(None, norm1, norm2).ratio()

def get_best_match(target, candidates, threshold=0.7):
    """
    Find the best match for the target string in the candidates list
    
    Args:
        target (str): The string to match
        candidates (list): List of candidate strings to match against
        threshold (float): Minimum similarity threshold (0-1)
        
    Returns:
        tuple: (best_match, similarity_score) or (None, 0) if no match found
    """
    if not target or not candidates:
        return None, 0
    
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        if candidate is None:
            continue
            
        # Skip empty strings or non-string values
        if not isinstance(candidate, str) or not candidate.strip():
            continue
            
        score = calculate_similarity(target, candidate)
        
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        return best_match, best_score
    
    return None, 0

def identify_statement_type(pdf_data):
    """
    Identify the type of financial statement from the PDF data
    
    Args:
        pdf_data (dict): Extracted PDF data
        
    Returns:
        str: Type of financial statement (balance_sheet, income_statement, changes_in_equity, etc.)
    """
    # Check document metadata first
    for key_value in pdf_data.get('key_value_pairs', []):
        if key_value.get('key') == 'document_type':
            document_type = key_value.get('value', '')
            if 'balance sheet' in document_type.lower():
                return 'balance_sheet'
            elif 'income statement' in document_type.lower():
                return 'income_statement'
            elif 'equity' in document_type.lower() and 'statement' in document_type.lower():
                return 'changes_in_equity'
            elif 'cash flow' in document_type.lower():
                return 'cash_flow'
    
    # Check for characteristic items in key-value pairs
    balance_sheet_indicators = ['total assets', 'liabilities', 'equity', 'current assets', 'non-current assets']
    income_statement_indicators = ['revenue', 'net income', 'profit', 'loss', 'expenses', 'earnings']
    equity_indicators = ['changes in equity', 'changes in members equity', 'total equity', 'retained earnings']
    cash_flow_indicators = ['cash flow', 'operating activities', 'investing activities', 'financing activities']
    
    indicators = {
        'balance_sheet': 0,
        'income_statement': 0,
        'changes_in_equity': 0,
        'cash_flow': 0
    }
    
    # Go through key-value pairs and look for indicator terms
    for kv in pdf_data.get('key_value_pairs', []):
        key = kv.get('key', '').lower()
        
        for indicator in balance_sheet_indicators:
            if indicator in key:
                indicators['balance_sheet'] += 1
                
        for indicator in income_statement_indicators:
            if indicator in key:
                indicators['income_statement'] += 1
                
        for indicator in equity_indicators:
            if indicator in key:
                indicators['changes_in_equity'] += 1
                
        for indicator in cash_flow_indicators:
            if indicator in key:
                indicators['cash_flow'] += 1
    
    # Find the statement type with the most indicators
    max_indicators = 0
    statement_type = 'unknown'
    
    for st_type, count in indicators.items():
        if count > max_indicators:
            max_indicators = count
            statement_type = st_type
    
    return statement_type

def map_data_with_openai(financial_data, sheet_structure, api_key=None):
    """
    Use OpenAI to map extracted financial data to sheet structure
    
    Args:
        financial_data (dict): Extracted financial data from PDF
        sheet_structure (dict): Structure of the Google Sheet
        api_key (str, optional): OpenAI API key
        
    Returns:
        dict: Mapping of cell references to values from PDF
    """
    import openai
    import json
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Check for API key
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
    
    openai.api_key = api_key
    
    # Log the full available data for debugging
    logger.info(f"Processing financial data with {len(financial_data.get('tables', []))} tables and {len(financial_data.get('key_value_pairs', []))} key-value pairs")
    
    # Prepare the input data: extract key parts to avoid token limits
    tables_data = []
    for table in financial_data.get('tables', []):
        table_info = {
            'type': table.get('type', 'Unknown'),
            'headers': table.get('headers', []),
            'data': table.get('data', [])  # Include all rows for better context
        }
        tables_data.append(table_info)
    
    # Identify the document type for better mapping
    document_type = "unknown"
    statement_date = "unknown"
    company_name = "unknown"
    
    for kv in financial_data.get('key_value_pairs', []):
        key = kv.get('key', '').lower()
        value = kv.get('value', '')
        
        if 'document_type' in key or 'statement_type' in key:
            document_type = value
        elif 'date' in key or 'period' in key or 'as of' in key or 'year end' in key:
            statement_date = value
        elif 'company' in key or 'organization' in key or 'entity' in key:
            company_name = value
    
    # Simplify sheet structure for the prompt and include the actual matrix data
    sheets_info = []
    for sheet in sheet_structure.get('sheets', []):
        # Include actual matrix data from the sheet if available
        matrix_data = sheet.get('values', [])
        
        # Don't limit the matrix size for better context
        matrix_preview = matrix_data
        
        sheet_info = {
            'title': sheet.get('title', ''),
            'headers': sheet.get('headers', []),
            'row_headers': sheet.get('rows_as_headers', []),
            'is_financial_statement': sheet.get('is_financial_statement', False),
            'matrix': matrix_preview  # Include the full matrix data
        }
        sheets_info.append(sheet_info)
    
    # Log the matrix data for sheets
    for sheet in sheets_info:
        matrix = sheet.get('matrix', [])
        logger.info(f"Including matrix for sheet '{sheet['title']}': {len(matrix)}x{len(matrix[0]) if matrix and matrix else 0}")
    
    # Prepare input JSON with additional context
    input_data = {
        'document_type': document_type,
        'statement_date': statement_date,
        'company_name': company_name,
        'key_value_pairs': financial_data.get('key_value_pairs', []),
        'tables': tables_data,
        'sheets': sheets_info,
        'metadata': financial_data.get('metadata', {})
    }
    
    # Prepare the prompt for mapping
    system_prompt = """You are a financial data mapping expert who specializes in precisely mapping financial data from PDFs into spreadsheet templates.
You understand financial statements including balance sheets, income statements, cash flow statements, and equity statements.
You can accurately map numbers, dates, and other data points to the appropriate cells in structured spreadsheets.
You always generate valid, well-formed JSON representing your mapping decisions."""
    
    prompt = f"""
# Financial Data Mapping Task

## Overview
You need to map extracted financial data from a PDF document into an existing Google Sheet template structure.

## Document Information
- Document Type: {document_type}
- Statement Date: {statement_date}
- Company: {company_name}

## Data Sources
The extracted financial data includes:
- {len(financial_data.get('key_value_pairs', []))} key-value pairs (metadata, important values, etc.)
- {len(financial_data.get('tables', []))} tables with structured data
- Document metadata

## Target Sheets
The Google Sheet template contains {len(sheets_info)} sheets, each with specific structure for financial data.

## IMPORTANT MAPPING INSTRUCTIONS
1. EXAMINE the Google Sheet matrices carefully - they contain the template structure where data should be mapped
2. For each cell in the template that needs data, FIND the corresponding value in the extracted financial data
3. MATCH values by looking at row labels and column headers in the sheet matrix
4. PRESERVE all existing values in the sheet - only suggest updates for empty cells or placeholders
5. MAP financial values to the most appropriate cells based on:
   - Row labels (item names like "Revenue", "Total Assets", etc.)
   - Column headers (time periods like "2023", "Dec 31", "Q4", etc.)
   - Financial statement structure (balance sheets, income statements, etc.)
6. DO NOT create cell updates for cells that already have data (unless they contain placeholders like "N/A" or "TBD")
7. USE exact cell references (like "A1", "B5", etc.) for each update
8. INCLUDE the source of each value for traceability

## DETAILED DATA
{json.dumps(input_data, indent=2)}

## RESPONSE FORMAT
Return ONLY a properly formatted JSON object with exactly this structure:

{{
  "updates": [
    {{
      "sheet_name": "Sheet Name", 
      "range": "A1",
      "value": "Value from extracted data",
      "source": "Description of where this data came from"
    }},
    ... more updates ...
  ],
  "new_sheets": [
    {{
      "title": "Generated Sheet Name",
      "headers": ["Column 1", "Column 2", ...],
      "data": [
        ["Row 1, Cell 1", "Row 1, Cell 2", ...],
        ... more rows ...
      ]
    }},
    ... more new sheets if needed ...
  ]
}}

DO NOT include any explanations or text outside of this JSON structure.
"""

    try:
        # Call OpenAI API for mapping using newer model with larger context window
        logger.info("Calling OpenAI for mapping with enhanced prompt")
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using model with largest context window
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistency
            max_tokens=12000,  # Increased max tokens for full context
            response_format={"type": "json_object"}  # Force JSON response format
        )
        
        # Extract the response content
        content = response.choices[0].message['content'].strip()
        logger.info(f"Received response from OpenAI (length: {len(content)})")
        
        try:
            # Parse the JSON directly - the response should be pure JSON
            mapping_result = json.loads(content)
            
            # Validate the structure
            if "updates" not in mapping_result:
                mapping_result["updates"] = []
            if "new_sheets" not in mapping_result:
                mapping_result["new_sheets"] = []
                
            # Log summary of mapping result
            update_count = len(mapping_result.get("updates", []))
            new_sheet_count = len(mapping_result.get("new_sheets", []))
            logger.info(f"Successfully generated mapping with {update_count} cell updates and {new_sheet_count} new sheets")
            
            # Log sheet-specific updates
            sheet_updates = {}
            for update in mapping_result.get("updates", []):
                sheet_name = update.get("sheet_name", "Unknown")
                if sheet_name not in sheet_updates:
                    sheet_updates[sheet_name] = 0
                sheet_updates[sheet_name] += 1
            
            for sheet_name, count in sheet_updates.items():
                logger.info(f"Sheet '{sheet_name}': {count} updates")
                
            return mapping_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.debug(f"Response content (preview): {content[:500]}...")
            
            # Try to extract JSON with regex as fallback
            import re
            json_match = re.search(r'(\{.*\})', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                try:
                    # Try to fix common JSON issues
                    # Replace single quotes with double quotes
                    json_str = json_str.replace("'", '"')
                    # Remove trailing commas in arrays and objects
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    
                    mapping_result = json.loads(json_str)
                    logger.info("Successfully extracted and fixed JSON using regex")
                    
                    # Validate the structure
                    if "updates" not in mapping_result:
                        mapping_result["updates"] = []
                    if "new_sheets" not in mapping_result:
                        mapping_result["new_sheets"] = []
                        
                    return mapping_result
                except json.JSONDecodeError as e2:
                    logger.error(f"Failed to parse JSON even after fixing: {e2}")
            
            # Return a basic structure with the error
            return {
                "error": f"Failed to parse OpenAI response: {e}",
                "raw_response": content[:1000],  # Include part of the response for debugging
                "updates": []
            }
    except Exception as e:
        logger.error(f"Error calling OpenAI API for mapping: {e}")
        return {
            "error": f"OpenAI API error: {e}",
            "updates": []
        }

def map_pdf_data_to_sheet(financial_data, sheet_structure, use_openai=True, api_key=None):
    """
    Map PDF data to sheet structure
    
    Args:
        financial_data (dict): Extracted financial data from PDF
        sheet_structure (dict): Structure of the Google Sheet
        use_openai (bool): Whether to use OpenAI for mapping
        api_key (str, optional): OpenAI API key
        
    Returns:
        dict: Mapping of cell references to values from PDF
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # First verify that the sheet structure contains matrix data
    has_matrix_data = False
    for sheet in sheet_structure.get('sheets', []):
        if 'values' in sheet and sheet['values']:
            has_matrix_data = True
            logger.info(f"Sheet '{sheet.get('title')}' has matrix data: {len(sheet['values'])}x{len(sheet['values'][0]) if sheet['values'] else 0}")
    
    if not has_matrix_data:
        logger.warning("No matrix data found in any sheets. Mapping may not be as accurate.")
    
    if use_openai:
        try:
            # Use OpenAI for mapping
            logger.info("Using OpenAI for mapping financial data to sheet structure")
            mapping_result = map_data_with_openai(financial_data, sheet_structure, api_key)
            
            # Check if we got a valid mapping
            if mapping_result and "updates" in mapping_result and mapping_result["updates"]:
                update_count = len(mapping_result["updates"])
                logger.info(f"Successfully mapped {update_count} cells using OpenAI")
                
                # Summarize which sheets were updated
                sheets_updated = set()
                for update in mapping_result.get("updates", []):
                    sheets_updated.add(update.get("sheet_name", "Unknown"))
                logger.info(f"Updated sheets: {', '.join(sheets_updated)}")
                
                return mapping_result
            else:
                logger.warning("OpenAI mapping failed or returned empty result, falling back to rule-based mapping")
        except Exception as e:
            logger.error(f"Error with OpenAI mapping: {e}")
            logger.info("Falling back to rule-based mapping")
    
    # Fall back to traditional rule-based mapping
    logger.info("Using rule-based mapping")
    
    # This is a placeholder. In a real implementation, you would have your existing mapping code here.
    # For now we'll return a basic structure
    basic_mapping = {
        "updates": [],
        "new_sheets": []
    }
    
    # Try to map key-value pairs to cells
    for sheet in sheet_structure.get('sheets', []):
        sheet_name = sheet.get('title', '')
        if not sheet_name:
            continue
            
        # For each key-value pair, try to find a matching cell
        for kv in financial_data.get('key_value_pairs', []):
            key = kv.get('key', '').lower()
            value = kv.get('value', '')
            
            # Skip empty values
            if not value:
                continue
                
            # Look for matching row headers in the sheet
            row_headers = sheet.get('rows_as_headers', [])
            for i, header in enumerate(row_headers):
                if not header:
                    continue
                    
                header_lower = str(header).lower()
                
                # Check for similarity
                if key in header_lower or header_lower in key:
                    # Found a match, add an update for column B (common structure)
                    basic_mapping["updates"].append({
                        "sheet_name": sheet_name,
                        "range": f"B{i+1}",  # A1 notation
                        "value": value,
                        "source": f"Key-value pair: {kv.get('key')}"
                    })
                    break
    
    logger.info(f"Rule-based mapping created {len(basic_mapping['updates'])} cell updates")
    return basic_mapping

def process_query(query, pdf_data, sheet_structure):
    """
    Process a natural language query about the financial data
    
    Args:
        query (str): Natural language query
        pdf_data (dict): Extracted data from PDF
        sheet_structure (dict): Structure of Google Sheet
        
    Returns:
        dict: Query response
    """
    # Check for common query patterns
    # 1. Looking for a specific value
    value_pattern = re.compile(r'(what|how much|find|get|show)\s+(?:is|are|was|were)?\s+(?:the)?\s+([^?]+)')
    value_match = value_pattern.search(query.lower())
    
    if value_match:
        search_term = value_match.group(2).strip()
        
        # Look for matching items in key-value pairs
        matches = []
        for kv in pdf_data.get('key_value_pairs', []):
            key = kv.get('key', '').lower()
            similarity = calculate_similarity(search_term, key)
            
            if similarity > 0.7:
                matches.append({
                    'key': kv.get('key'),
                    'value': kv.get('value'),
                    'similarity': similarity
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        if matches:
            return {
                'query_type': 'value_lookup',
                'search_term': search_term,
                'matches': matches,
                'response': f"Found {len(matches)} matches for '{search_term}'"
            }
        else:
            return {
                'query_type': 'value_lookup',
                'search_term': search_term,
                'matches': [],
                'response': f"No matches found for '{search_term}'"
            }
    
    # 2. Fill a specific cell or update the sheet
    fill_pattern = re.compile(r'(fill|update|put|set|write)\s+(?:the)?\s+([^?]+?)\s+(?:in|into|to|with|as)\s+(?:the)?\s+([^?]+)')
    fill_match = fill_pattern.search(query.lower())
    
    if fill_match:
        value_term = fill_match.group(2).strip()
        location_term = fill_match.group(3).strip()
        
        # Look for matching values
        value_matches = []
        for kv in pdf_data.get('key_value_pairs', []):
            key = kv.get('key', '').lower()
            similarity = calculate_similarity(value_term, key)
            
            if similarity > 0.6:
                value_matches.append({
                    'key': kv.get('key'),
                    'value': kv.get('value'),
                    'numeric_value': kv.get('numeric_value'),
                    'similarity': similarity
                })
        
        # Sort by similarity
        value_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        if not value_matches:
            return {
                'query_type': 'update_request',
                'value_term': value_term,
                'location_term': location_term,
                'matched_value': None,
                'matched_location': None,
                'response': f"Could not find any matches for '{value_term}'"
            }
        
        # Choose the best value match
        best_value_match = value_matches[0]
        
        # Now look for matching locations in the sheet
        location_matches = []
        
        for sheet in sheet_structure.get('sheets', []):
            sheet_title = sheet.get('title', '')
            
            # Check if the location term is the sheet name
            sheet_similarity = calculate_similarity(location_term, sheet_title)
            
            if sheet_similarity > 0.7:
                # Match to the first cell that makes sense
                # For financial statements, usually this would be putting a value in column B (second column)
                location_matches.append({
                    'sheet': sheet_title,
                    'cell': 'B2',  # Default to B2 as a reasonable target
                    'similarity': sheet_similarity,
                    'match_type': 'sheet'
                })
                continue
            
            # Check rows (for transposed financial statements)
            row_headers = [row[0] if row and len(row) > 0 else "" for row in sheet.get('values', [])]
            
            for i, header in enumerate(row_headers):
                if not header or not isinstance(header, str):
                    continue
                    
                similarity = calculate_similarity(location_term, header)
                
                if similarity > 0.7:
                    # For transposed sheets, typically the second column has values
                    col_letter = 'B'  # Default to column B (second column)
                    
                    location_matches.append({
                        'sheet': sheet_title,
                        'cell': f"{col_letter}{i+1}",
                        'similarity': similarity,
                        'match_type': 'row'
                    })
        
        # Sort by similarity
        location_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        if not location_matches:
            return {
                'query_type': 'update_request',
                'value_term': value_term,
                'location_term': location_term,
                'matched_value': best_value_match,
                'matched_location': None,
                'response': f"Found value '{best_value_match['key']}' = '{best_value_match['value']}' but could not find a matching location for '{location_term}'"
            }
        
        # Choose the best location match
        best_location_match = location_matches[0]
        
        # Prepare update
        update = {
            'sheet_name': best_location_match['sheet'],
            'range': best_location_match['cell'],
            'value': best_value_match['numeric_value'] if best_value_match['numeric_value'] is not None else best_value_match['value']
        }
        
        return {
            'query_type': 'update_request',
            'value_term': value_term,
            'location_term': location_term,
            'matched_value': best_value_match,
            'matched_location': best_location_match,
            'update': update,
            'response': f"Will update cell {best_location_match['cell']} in sheet '{best_location_match['sheet']}' with value '{best_value_match['value']}'"
        }
    
    # If no patterns match, return a generic response
    return {
        'query_type': 'unknown',
        'response': f"I'm not sure how to handle this query: '{query}'. Try asking for a specific value or asking to fill a specific cell."
    } 