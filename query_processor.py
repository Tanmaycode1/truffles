import json
import re
from mapping_engine import process_query, get_best_match, calculate_similarity
from sheets_manager import update_sheet_with_data
import openai
import os

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