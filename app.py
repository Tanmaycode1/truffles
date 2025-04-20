import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
import uuid
import json
from dotenv import load_dotenv
import shutil
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from pdf_processor import extract_financial_data_from_pdf, detect_document_type
from sheets_manager import get_sheet_structure, extract_spreadsheet_id, create_new_sheet, test_google_sheets_access, get_credentials
# Import simple_mapper functionality only
from simple_mapper import map_financial_data, save_context_to_file
# Import query processor functionality
from query_processor import QuerySession, process_query, execute_action

load_dotenv()

# Ensure credentials are set up
def setup_google_credentials():
    """Set up Google API credentials from the credentials.json file"""
    # Define the paths
    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(app_dir)
    source_credentials = os.path.join(project_dir, 'credentials.json')
    target_credentials = 'credentials.json'  # In the working directory
    
    # Copy credentials.json to the right location if it doesn't exist
    if os.path.exists(source_credentials) and not os.path.exists(target_credentials):
        shutil.copy(source_credentials, target_credentials)
        print(f"Copied credentials.json to {target_credentials}")
        
    # If source file doesn't exist but the one in app folder does exist, all good    
    elif os.path.exists(target_credentials):
        print(f"credentials.json already exists at {target_credentials}")
    
    # Neither source nor target exist    
    else:
        print("Warning: credentials.json not found. Google Sheets integration may not work.")
    
    # Test Google Sheets access
    test_google_sheets_access()

def test_google_sheets_access():
    """Test if Google Sheets API is accessible with current credentials."""
    try:
        # Get credentials and build the Sheets API service
        creds = get_credentials()
        if not creds:
            logging.warning("No valid Google credentials found")
            return False
            
        service = build('sheets', 'v4', credentials=creds)
        
        # Make a simple API call to test access
        # Just get the spreadsheet metadata for a dummy spreadsheet ID
        # This will fail with a 404, but that's expected - we just want to check auth
        try:
            service.spreadsheets().get(spreadsheetId="1").execute()
        except HttpError as error:
            # 404 error is expected since we're using a dummy ID
            if error.resp.status == 404:
                return True
            else:
                logging.warning(f"Google Sheets API error: {error}")
                return False
        
        return True
    except Exception as e:
        logging.error(f"Error testing Google Sheets access: {e}")
        return False

# Run setup
setup_google_credentials()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Check if Google Sheets access is working
    google_sheets_working = test_google_sheets_access()
    return render_template('index.html', google_sheets_working=google_sheets_working)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['pdf_file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = str(uuid.uuid4()) + '.pdf'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the PDF with OpenAI for key-value extraction
        try:
            # Get OpenAI API key from environment
            api_key = os.environ.get('OPENAI_API_KEY')
            print("Processing PDF file...")
            pdf_data = extract_financial_data_from_pdf(filepath, api_key=api_key)
            print(f"PDF processed: {len(pdf_data.get('key_value_pairs', []))} key-value pairs, {len(pdf_data.get('tables', []))} tables")
            
            # Create a compressed version of the data for the session
            session_data = {
                'metadata': pdf_data.get('metadata', {}),
                'file_path': filepath,
                'table_count': len(pdf_data.get('tables', [])),
                'kv_count': len(pdf_data.get('key_value_pairs', [])),
                'timestamp': str(uuid.uuid4())  # Add a unique id to identify this extraction
            }
            
            # Store the session summary and filepath
            session['pdf_data_summary'] = json.dumps(session_data)
            session['pdf_filepath'] = filepath
            
            # Store the full data in a file to avoid session size limits
            data_filename = f"{os.path.splitext(filename)[0]}_data.json"
            data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            with open(data_filepath, 'w') as f:
                json.dump(pdf_data, f)
            
            # Store the data filepath in the session
            session['pdf_data_filepath'] = data_filepath
            
            return jsonify({
                'success': True,
                'message': 'PDF processed successfully',
                'data': pdf_data
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

# Helper function to load PDF data
def load_pdf_data():
    """Load PDF data from saved JSON file or extract from PDF if needed"""
    if 'pdf_data_filepath' in session and os.path.exists(session['pdf_data_filepath']):
        # Load data from JSON file
        try:
            with open(session['pdf_data_filepath'], 'r') as f:
                print(f"Loading PDF data from cached file: {session['pdf_data_filepath']}")
                return json.load(f)
        except Exception as e:
            print(f"Error loading cached PDF data: {e}")
    
    # Fall back to processing the PDF file directly
    if 'pdf_filepath' in session and os.path.exists(session['pdf_filepath']):
        print(f"Re-processing PDF file: {session['pdf_filepath']}")
        api_key = os.environ.get('OPENAI_API_KEY')
        return extract_financial_data_from_pdf(session['pdf_filepath'], api_key=api_key)
    
    raise ValueError("No PDF data found. Please upload a PDF first.")

@app.route('/get-sheet-structure', methods=['GET'])
def get_sheet_structure_endpoint():
    """Get the structure of a Google Sheet"""
    sheet_url = request.args.get('url')
    if not sheet_url:
        return jsonify({'success': False, 'error': 'Google Sheet URL is required'}), 400
    
    print(f"Attempting to get structure for sheet URL: {sheet_url}")
    
    try:
        # Get sheet structure
        structure = get_sheet_structure(sheet_url)
        
        # Process the structure to include sample matrix data for display
        for sheet in structure.get('sheets', []):
            # Ensure we have the values (matrix data)
            if 'values' in sheet:
                # Include up to 10 rows of matrix data for display
                sheet['matrix_preview'] = sheet['values'][:10] if sheet['values'] else []
                
                # Calculate matrix dimensions
                sheet['matrix_rows'] = len(sheet['values'])
                sheet['matrix_cols'] = max([len(row) for row in sheet['values']]) if sheet['values'] else 0
            else:
                sheet['matrix_preview'] = []
                sheet['matrix_rows'] = 0
                sheet['matrix_cols'] = 0
        
        print(f"Successfully retrieved sheet structure. Title: {structure.get('title')}, Sheets: {len(structure.get('sheets', []))}")
        
        return jsonify({
            'success': True,
            'structure': structure
        })
    except Exception as e:
        print(f"Error getting sheet structure: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/preview-mapping', methods=['POST'])
def preview_mapping():
    """Preview how data will be mapped across all sheets"""
    if 'pdf_filepath' not in session:
        return jsonify({'error': 'No PDF data found. Please upload a PDF first.'}), 400
    
    sheet_url = request.form.get('sheet_url')
    if not sheet_url:
        return jsonify({'error': 'Google Sheet URL is required'}), 400
    
    try:
        # Get sheet structure for all tabs
        sheet_structure = get_sheet_structure(sheet_url)
        
        # Load PDF data from cached file
        pdf_data = load_pdf_data()
        
        # Check if we have sheets to process
        if not sheet_structure.get('sheets'):
            return jsonify({'error': 'No sheets found in the spreadsheet'}), 400
        
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
                mapping_result = map_financial_data(pdf_data, matrix)
                
                # Add to results
                all_mapping_results.append({
                    'sheet_name': sheet_name,
                    'filled_matrix': mapping_result.get('filled_matrix', []),
                    'stats': mapping_result.get('stats', {}),
                    'status': 'success'
                })
            except Exception as e:
                all_mapping_results.append({
                    'sheet_name': sheet_name,
                    'error': str(e),
                    'status': 'error'
                })
        
        # Store the mapping results in the session for later use
        session['all_mapping_results'] = json.dumps(all_mapping_results)
        
        # Format the result for the preview
        result = {
            'success': True,
            'mapping_results': all_mapping_results,
            'spreadsheet_id': extract_spreadsheet_id(sheet_url),
            'spreadsheet_title': sheet_structure.get('title', 'Untitled Spreadsheet')
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/map-to-sheet', methods=['POST'])
def map_to_sheet():
    """Migrate the filled matrices to Google Sheets"""
    if 'pdf_filepath' not in session or 'all_mapping_results' not in session:
        return jsonify({'error': 'No mapping data found. Please preview the mapping first.'}), 400
    
    sheet_url = request.form.get('sheet_url')
    if not sheet_url:
        return jsonify({'error': 'Google Sheet URL is required'}), 400
    
    # Check if we should create new sheets
    create_new = request.form.get('create_new', 'false') == 'true'
    
    # Get selected sheets to migrate
    selected_sheets = request.form.getlist('selected_sheets')
    if not selected_sheets:
        return jsonify({'error': 'No sheets selected for migration'}), 400
    
    try:
        # Get the mapping results from the session
        all_mapping_results = json.loads(session['all_mapping_results'])
        
        # Filter to only include selected sheets
        migration_results = []
        
        # Get spreadsheet ID
        spreadsheet_id = extract_spreadsheet_id(sheet_url)
        
        # Initialize Google Sheets API
        from googleapiclient.discovery import build
        from sheets_manager import get_credentials
        
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)
        
        # Process each selected sheet
        for sheet_result in all_mapping_results:
            sheet_name = sheet_result.get('sheet_name')
            
            # Skip sheets that weren't selected
            if sheet_name not in selected_sheets:
                continue
                
            # Skip sheets that had errors
            if sheet_result.get('status') != 'success':
                migration_results.append({
                    'sheet_name': sheet_name,
                    'status': 'skipped',
                    'message': f"Sheet was skipped due to previous errors: {sheet_result.get('error', 'Unknown error')}"
                })
                continue
            
            filled_matrix = sheet_result.get('filled_matrix', [])
            stats = sheet_result.get('stats', {})
            
            if create_new:
                # Generate a unique name for the new sheet
                timestamp = uuid.uuid4().hex[:8]
                new_sheet_name = f"{sheet_name}_filled_{timestamp}"
                
                # Create a new sheet with the filled matrix
                result = create_new_sheet(
                    service,
                    spreadsheet_id,
                    new_sheet_name,
                    headers=None,  # No separate headers, everything is in the matrix
                    data=filled_matrix
                )
                
                migration_results.append({
                    'sheet_name': sheet_name,
                    'new_sheet_name': new_sheet_name,
                    'status': 'created',
                    'stats': stats,
                    'message': f'Created new sheet: {new_sheet_name}'
                })
            else:
                # REPLACE the existing sheet with the filled matrix
                try:
                    # First, clear the entire sheet
                    clear_request = service.spreadsheets().values().clear(
                        spreadsheetId=spreadsheet_id,
                        range=f"{sheet_name}"
                    )
                    clear_request.execute()
                    
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
                    
                    migration_results.append({
                        'sheet_name': sheet_name,
                        'status': 'replaced',
                        'stats': stats,
                        'message': f'Replaced sheet: {sheet_name}'
                    })
                except Exception as e:
                    migration_results.append({
                        'sheet_name': sheet_name,
                        'status': 'error',
                        'message': f'Error updating sheet {sheet_name}: {str(e)}'
                    })
        
        return jsonify({
            'success': True,
            'message': f'Migration completed for {len(migration_results)} sheets',
            'migration_results': migration_results,
            'spreadsheet_id': spreadsheet_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create-sheet-template', methods=['POST'])
def create_sheet_template():
    """Create a new sheet with a template based on a PDF structure"""
    if 'pdf_filepath' not in session:
        return jsonify({'error': 'No PDF data found. Please upload a PDF first.'}), 400
    
    sheet_url = request.form.get('sheet_url')
    if not sheet_url:
        return jsonify({'error': 'Google Sheet URL is required'}), 400
    
    template_name = request.form.get('template_name', 'Financial Template')
    
    try:
        # Load PDF data from cached file
        pdf_data = load_pdf_data()
        
        # Extract spreadsheet ID
        spreadsheet_id = extract_spreadsheet_id(sheet_url)
        
        # Get credentials and build service
        from googleapiclient.discovery import build
        from sheets_manager import get_credentials
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)
        
        # Generate template headers based on PDF data
        template_headers = ["Item"]
        template_rows = []
        
        # Add a date column based on statement date if found
        statement_date = None
        for kv in pdf_data.get('key_value_pairs', []):
            if kv.get('key') == 'statement_date':
                statement_date = kv.get('value')
                template_headers.append(statement_date)
                break
        
        if not statement_date:
            template_headers.append("Value")
        
        # Add rows for each key value pair
        for kv in pdf_data.get('key_value_pairs', []):
            key = kv.get('key')
            value = kv.get('value', '')
            
            # Skip metadata fields that would be redundant
            if key in ['document_type', 'statement_date', 'company_name']:
                continue
                
            template_rows.append([key, value])
        
        # Sort rows alphabetically for better organization
        template_rows.sort(key=lambda row: row[0])
        
        # Create the template sheet
        result = create_new_sheet(
            service, 
            spreadsheet_id, 
            template_name, 
            headers=template_headers, 
            data=template_rows
        )
        
        return jsonify({
            'success': True,
            'message': f'Created template sheet: {template_name}',
            'sheet_result': result,
            'spreadsheet_id': spreadsheet_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def handle_query():
    """Handle a query about the data or Google Sheet"""
    if 'pdf_filepath' not in session:
        return jsonify({'error': 'No PDF data found. Please upload a PDF first.'}), 400
    
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    sheet_url = request.form.get('sheet_url')
    sheet_name = request.form.get('sheet_name')
    session_id = request.form.get('session_id')
        
    try:
        # Load PDF data from cached file
        pdf_data = load_pdf_data()
        
        # Get mapping results if available
        filled_matrix = None
        if 'all_mapping_results' in session:
            mapping_results = json.loads(session['all_mapping_results'])
            # If sheet_name is specified, find that specific sheet's filled matrix
            if sheet_name:
                for result in mapping_results:
                    if result.get('sheet_name') == sheet_name and result.get('status') == 'success':
                        filled_matrix = result.get('filled_matrix')
                        break
            # Otherwise use the first successful mapping
            else:
                for result in mapping_results:
                    if result.get('status') == 'success':
                        filled_matrix = result.get('filled_matrix')
                        sheet_name = result.get('sheet_name')
                        break
        
        # Get or create a query session
        query_session = None
        spreadsheet_id = None
        
        if sheet_url:
            spreadsheet_id = extract_spreadsheet_id(sheet_url)
        
        # Check if we have an existing session stored
        if 'query_session' in session and session_id:
            stored_session = json.loads(session['query_session'])
            if stored_session.get('session_id') == session_id:
                # Recreate the session with current data
                query_session = QuerySession.from_dict(
                    stored_session,
                    financial_data=pdf_data,
                    filled_matrix=filled_matrix
                )
        
        # Create a new session if needed
        if not query_session:
            query_session = QuerySession(
                financial_data=pdf_data,
                filled_matrix=filled_matrix,
                sheet_url=sheet_url,
                sheet_name=sheet_name
            )
        
        # Process the query
        result = process_query(query_session, query)
        
        # Execute any actions if present
        action_result = None
        if result.get('action'):
            action_result = execute_action(
                query_session,
                result.get('action'),
                spreadsheet_id=spreadsheet_id
            )
        
        # Store the session for future queries
        session['query_session'] = json.dumps(query_session.to_dict())
        
        # Return the response
        response = {
            'success': True,
            'query': query,
            'response': result.get('response'),
            'session_id': query_session.session_id
        }
        
        if action_result:
            response['action_result'] = action_result
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# New route for saving the context to a file for debugging
@app.route('/save-context', methods=['POST'])
def save_context():
    """Save the formatted context to a file for debugging"""
    if 'pdf_filepath' not in session:
        return jsonify({'error': 'No PDF data found. Please upload a PDF first.'}), 400
    
    try:
        # Load PDF data from cached file
        pdf_data = load_pdf_data()
        
        # Generate a filename for the context
        context_filename = f"context_{uuid.uuid4().hex[:8]}.txt"
        context_filepath = os.path.join(app.config['UPLOAD_FOLDER'], context_filename)
        
        # Save the context to a file
        result = save_context_to_file(pdf_data, context_filepath)
        
        if result:
            return jsonify({
                'success': True,
                'message': 'Context saved successfully',
                'filepath': context_filepath
            })
        else:
            return jsonify({'error': 'Failed to save context'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 