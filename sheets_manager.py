import os
import json
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import re
import pandas as pd

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

def get_credentials():
    """Get Google API credentials from saved token or service account"""
    creds = None
    
    # Check if token.json exists and has content
    if os.path.exists('token.json') and os.path.getsize('token.json') > 0:
        try:
            # Try to load from token.json
            with open('token.json', 'r') as token_file:
                token_data = json.load(token_file)
                
            # Check if it's a service account token
            if token_data.get('type') == 'service_account':
                # Load as service account
                creds = service_account.Credentials.from_service_account_info(
                    token_data, scopes=SCOPES)
            else:
                # Load as OAuth credentials
                creds = Credentials.from_authorized_user_info(token_data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error reading token.json: {str(e)}")
    
    # If there are no valid credentials available or token.json is invalid
    if not creds:
        # Check for service account first
        if os.path.exists('service_account.json'):
            try:
                creds = service_account.Credentials.from_service_account_file(
                    'service_account.json', scopes=SCOPES)
                
                # Save to token.json for next time
                import shutil
                shutil.copy('service_account.json', 'token.json')
                print("Copied service_account.json to token.json")
            except Exception as e:
                print(f"Error with service account: {str(e)}")
        else:
            # Fall back to OAuth flow if no service account
            if not os.path.exists('credentials.json'):
                raise FileNotFoundError(
                    "No credentials found. Please provide either credentials.json for OAuth or service_account.json")
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
    
    return creds

def extract_spreadsheet_id(sheet_url):
    """Extract the spreadsheet ID from a Google Sheets URL"""
    # Format: https://docs.google.com/spreadsheets/d/{spreadsheetId}/edit#gid=0
    match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url)
    if match:
        return match.group(1)
    return sheet_url  # If not a URL, assume it's already the spreadsheet ID

def get_sheet_structure(sheet_url):
    """
    Get the structure of a Google Sheet including sheets, headers, and cell formatting
    
    Args:
        sheet_url (str): URL of the Google Sheet or spreadsheet ID
        
    Returns:
        dict: Structure of the Google Sheet
    """
    spreadsheet_id = extract_spreadsheet_id(sheet_url)
    
    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)
        
        # Get spreadsheet metadata
        spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        
        sheet_structure = {
            'spreadsheet_id': spreadsheet_id,
            'title': spreadsheet.get('properties', {}).get('title', ''),
            'sheets': []
        }
        
        # Process each sheet
        for sheet in spreadsheet.get('sheets', []):
            properties = sheet.get('properties', {})
            sheet_id = properties.get('sheetId', 0)
            sheet_title = properties.get('title', '')
            
            # Detect if the sheet contains a financial statement based on name
            is_financial_statement = any(keyword in sheet_title.lower() for keyword in 
                                      ['income', 'balance', 'cash flow', 'equity', 'statement'])
            
            # Get the sheet data for analysis
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=f"'{sheet_title}'",
                valueRenderOption='UNFORMATTED_VALUE'
            ).execute()
            
            values = result.get('values', [])
            
            # Analyze the structure
            sheet_data = {
                'sheet_id': sheet_id,
                'title': sheet_title,
                'is_financial_statement': is_financial_statement,
                'rows': len(values),
                'cols': max([len(row) for row in values]) if values else 0,
                'headers': [],
                'rows_as_headers': []
            }
            
            # Try to identify headers (first row often contains headers)
            if values and len(values) > 0:
                sheet_data['headers'] = values[0] if values[0] else []
                
                # Check if rows have labels in first column (common in financial statements)
                # Where the structure is transposed (rows become columns)
                row_headers = []
                for row in values:
                    if row and len(row) > 0:
                        row_headers.append(row[0])
                
                sheet_data['rows_as_headers'] = row_headers
                
                # Check if this appears to be a transposed sheet (financial statements often have this structure)
                # Where line items are in rows and periods/years are in columns
                non_empty_first_cells = sum(1 for row in values if row and len(row) > 0 and row[0])
                sheet_data['appears_transposed'] = non_empty_first_cells > len(sheet_data['headers']) / 2
                
                # Save the raw values for deeper analysis
                sheet_data['values'] = values
            
            sheet_structure['sheets'].append(sheet_data)
        
        return sheet_structure
    
    except Exception as e:
        raise Exception(f"Error getting sheet structure: {str(e)}")

def update_sheet_with_data(sheet_url, mapping_result):
    """
    Update a Google Sheet with mapped data
    
    Args:
        sheet_url (str): URL of the Google Sheet or spreadsheet ID
        mapping_result (dict): Mapping of sheet cells to values from PDF
        
    Returns:
        dict: Result of the update operation
    """
    spreadsheet_id = extract_spreadsheet_id(sheet_url)
    
    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)
        
        # Prepare batch update request
        batch_update_values_request_body = {
            'value_input_option': 'USER_ENTERED',
            'data': []
        }
        
        # Process each update in the mapping result
        for update in mapping_result.get('updates', []):
            sheet_name = update.get('sheet_name', '')
            cell_range = update.get('range', '')
            value = update.get('value', '')
            
            # Format range for API
            full_range = f"'{sheet_name}'!{cell_range}"
            
            # Format value for API (must be 2D array)
            values = [[value]]
            
            batch_update_values_request_body['data'].append({
                'range': full_range,
                'values': values
            })
        
        # Skip if no updates
        if not batch_update_values_request_body['data']:
            return {'status': 'no_updates', 'message': 'No updates to process'}
        
        # Execute batch update
        result = service.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=batch_update_values_request_body
        ).execute()
        
        return {
            'status': 'success',
            'updated_cells': result.get('totalUpdatedCells', 0),
            'updated_sheets': result.get('totalUpdatedSheets', 0)
        }
    
    except Exception as e:
        raise Exception(f"Error updating sheet: {str(e)}")

def cell_ref_to_a1_notation(row, col):
    """
    Convert 0-based row and column indices to A1 notation
    
    Args:
        row (int): 0-based row index
        col (int): 0-based column index
        
    Returns:
        str: Cell reference in A1 notation
    """
    col_str = ''
    
    # Convert column index to letter(s)
    while col >= 0:
        col_str = chr(65 + (col % 26)) + col_str
        col = col // 26 - 1
        
    # Rows are 1-based in A1 notation
    return f"{col_str}{row + 1}"

def read_sheet_range(sheet_url, range_name):
    """
    Read a range of values from a Google Sheet
    
    Args:
        sheet_url (str): URL of the Google Sheet or spreadsheet ID
        range_name (str): A1 notation of the range to read
        
    Returns:
        list: 2D array of values
    """
    spreadsheet_id = extract_spreadsheet_id(sheet_url)
    
    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)
        
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueRenderOption='UNFORMATTED_VALUE'
        ).execute()
        
        return result.get('values', [])
    
    except Exception as e:
        raise Exception(f"Error reading sheet range: {str(e)}")

def create_new_sheet(service, spreadsheet_id, sheet_title, headers=None, data=None):
    """
    Create a new sheet in a Google Sheet
    
    Args:
        service: Google Sheets API service
        spreadsheet_id (str): ID of the spreadsheet
        sheet_title (str): Title of the new sheet
        headers (list, optional): List of column headers
        data (list, optional): 2D array of data rows
        
    Returns:
        dict: Result of the sheet creation
    """
    try:
        # Add new sheet request
        add_sheet_request = {
            'addSheet': {
                'properties': {
                    'title': sheet_title
                }
            }
        }
        
        # Execute request to add the sheet
        result = service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={'requests': [add_sheet_request]}
        ).execute()
        
        # Get the new sheet ID
        new_sheet_id = result['replies'][0]['addSheet']['properties']['sheetId']
        
        # If headers or data is provided, update the sheet
        if headers or data:
            values = []
            
            # Add headers if provided
            if headers:
                values.append(headers)
            
            # Add data if provided
            if data:
                values.extend(data)
            
            # Update the sheet with values
            if values:
                service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"'{sheet_title}'!A1",
                    valueInputOption="USER_ENTERED",
                    body={'values': values}
                ).execute()
        
        return {
            'status': 'success',
            'sheet_id': new_sheet_id,
            'title': sheet_title
        }
    
    except Exception as e:
        if "already exists" in str(e).lower():
            # Sheet already exists, get its ID
            result = service.spreadsheets().get(
                spreadsheetId=spreadsheet_id,
                fields='sheets.properties'
            ).execute()
            
            for sheet in result.get('sheets', []):
                properties = sheet.get('properties', {})
                if properties.get('title') == sheet_title:
                    # Clear the existing sheet and update with new data
                    if headers or data:
                        values = []
                        if headers:
                            values.append(headers)
                        if data:
                            values.extend(data)
                        
                        if values:
                            service.spreadsheets().values().clear(
                                spreadsheetId=spreadsheet_id,
                                range=f"'{sheet_title}'!A1:Z1000"  # Clear a large range
                            ).execute()
                            
                            service.spreadsheets().values().update(
                                spreadsheetId=spreadsheet_id,
                                range=f"'{sheet_title}'!A1",
                                valueInputOption="USER_ENTERED",
                                body={'values': values}
                            ).execute()
                    
                    return {
                        'status': 'updated_existing',
                        'sheet_id': properties.get('sheetId'),
                        'title': sheet_title
                    }
            
            # If we get here, something strange happened
            raise Exception(f"Sheet {sheet_title} reportedly exists but could not be found")
        else:
            raise Exception(f"Error creating sheet: {str(e)}")

def apply_mapping_result(sheet_url, mapping_result):
    """
    Apply a mapping result to a Google Sheet, including creating new sheets if needed
    
    Args:
        sheet_url (str): URL of the Google Sheet or spreadsheet ID
        mapping_result (dict): Mapping result from map_pdf_data_to_sheet
        
    Returns:
        dict: Result of the update operation
    """
    spreadsheet_id = extract_spreadsheet_id(sheet_url)
    
    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)
        
        result = {
            'status': 'success',
            'updated_cells': 0,
            'updated_sheets': 0,
            'new_sheets': []
        }
        
        # Create new sheets if specified
        if 'new_sheets' in mapping_result and mapping_result['new_sheets']:
            for sheet_spec in mapping_result['new_sheets']:
                title = sheet_spec.get('title')
                headers = sheet_spec.get('headers')
                data = sheet_spec.get('data')
                
                if title:
                    try:
                        sheet_result = create_new_sheet(service, spreadsheet_id, title, headers, data)
                        result['new_sheets'].append(sheet_result)
                        result['updated_sheets'] += 1
                    except Exception as e:
                        print(f"Error creating sheet {title}: {str(e)}")
        
        # Apply cell updates if specified
        if 'updates' in mapping_result and mapping_result['updates']:
            # Prepare batch update request
            batch_update_values_request_body = {
                'value_input_option': 'USER_ENTERED',
                'data': []
            }
            
            # Group updates by sheet name for more efficient updates
            updates_by_sheet = {}
            for update in mapping_result['updates']:
                sheet_name = update.get('sheet_name', '')
                cell_range = update.get('range', '')
                value = update.get('value', '')
                
                if not sheet_name or not cell_range:
                    continue
                
                if sheet_name not in updates_by_sheet:
                    updates_by_sheet[sheet_name] = []
                
                updates_by_sheet[sheet_name].append({
                    'range': cell_range,
                    'value': value
                })
            
            # Process updates for each sheet
            for sheet_name, updates in updates_by_sheet.items():
                # Check if the sheet exists
                try:
                    service.spreadsheets().values().get(
                        spreadsheetId=spreadsheet_id,
                        range=f"'{sheet_name}'!A1"
                    ).execute()
                except Exception:
                    # Sheet doesn't exist, create it
                    try:
                        create_new_sheet(service, spreadsheet_id, sheet_name)
                        result['new_sheets'].append({
                            'status': 'success',
                            'title': sheet_name
                        })
                        result['updated_sheets'] += 1
                    except Exception as e:
                        print(f"Error creating sheet {sheet_name}: {str(e)}")
                        continue
                
                # Prepare batch update for this sheet
                for update in updates:
                    cell_range = update['range']
                    value = update['value']
                    
                    # Format range for API
                    full_range = f"'{sheet_name}'!{cell_range}"
                    
                    # Format value for API (must be 2D array)
                    values = [[value]]
                    
                    batch_update_values_request_body['data'].append({
                        'range': full_range,
                        'values': values
                    })
            
            # Skip if no updates
            if not batch_update_values_request_body['data']:
                return result
            
            # Execute batch update
            update_result = service.spreadsheets().values().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body=batch_update_values_request_body
            ).execute()
            
            result['updated_cells'] = update_result.get('totalUpdatedCells', 0)
            if update_result.get('totalUpdatedSheets', 0) > 0:
                result['updated_sheets'] += update_result.get('totalUpdatedSheets', 0)
        
        return result
    
    except Exception as e:
        raise Exception(f"Error applying mapping result: {str(e)}")

if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) > 1:
        sheet_url = sys.argv[1]
        result = get_sheet_structure(sheet_url)
        print(json.dumps(result, indent=2)) 