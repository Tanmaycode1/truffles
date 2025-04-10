import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
import uuid
import json
from dotenv import load_dotenv

from pdf_processor import extract_financial_data_from_pdf
from sheets_manager import get_sheet_structure, apply_mapping_result, extract_spreadsheet_id, create_new_sheet
from mapping_engine import map_pdf_data_to_sheet

load_dotenv()

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
    return render_template('index.html')

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
    """Preview how data will be mapped without sending to sheets"""
    if 'pdf_filepath' not in session:
        return jsonify({'error': 'No PDF data found. Please upload a PDF first.'}), 400
    
    sheet_url = request.form.get('sheet_url')
    if not sheet_url:
        return jsonify({'error': 'Google Sheet URL is required'}), 400
    
    try:
        # Get sheet structure
        sheet_structure = get_sheet_structure(sheet_url)
        
        # Load PDF data from cached file
        pdf_data = load_pdf_data()
        
        # Map PDF data to sheet structure using OpenAI
        api_key = os.environ.get('OPENAI_API_KEY')
        mapping_result = map_pdf_data_to_sheet(pdf_data, sheet_structure, use_openai=True, api_key=api_key)
        
        return jsonify({
            'success': True,
            'mapping_preview': mapping_result,
            'spreadsheet_id': extract_spreadsheet_id(sheet_url)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/map-to-sheet', methods=['POST'])
def map_to_sheet():
    if 'pdf_filepath' not in session:
        return jsonify({'error': 'No PDF data found. Please upload a PDF first.'}), 400
    
    sheet_url = request.form.get('sheet_url')
    if not sheet_url:
        return jsonify({'error': 'Google Sheet URL is required'}), 400
    
    # Check if we should create a new sheet
    create_new = request.form.get('create_new', 'false') == 'true'
    
    try:
        # Get sheet structure
        sheet_structure = get_sheet_structure(sheet_url)
        
        # Load PDF data from cached file
        pdf_data = load_pdf_data()
        
        # Map PDF data to sheet structure using OpenAI
        api_key = os.environ.get('OPENAI_API_KEY')
        mapping_result = map_pdf_data_to_sheet(pdf_data, sheet_structure, use_openai=True, api_key=api_key)
        
        # If create_new is true, modify the mapping to create new sheets
        if create_new and 'updates' in mapping_result:
            # Group updates by sheet name
            updates_by_sheet = {}
            for update in mapping_result['updates']:
                sheet_name = update.get('sheet_name')
                if sheet_name not in updates_by_sheet:
                    updates_by_sheet[sheet_name] = []
                updates_by_sheet[sheet_name].append(update)
            
            # Create new sheets info
            new_sheets = []
            for sheet_name, updates in updates_by_sheet.items():
                # Create a new sheet name with timestamp
                timestamp = uuid.uuid4().hex[:8]
                new_sheet_name = f"{sheet_name}_filled_{timestamp}"
                
                # Add to new sheets list
                new_sheets.append({
                    'title': new_sheet_name,
                    'source_sheet': sheet_name,
                    'updates': updates
                })
            
            # Update mapping result to use new sheet names
            modified_updates = []
            for update in mapping_result['updates']:
                sheet_name = update.get('sheet_name')
                for new_sheet in new_sheets:
                    if new_sheet['source_sheet'] == sheet_name:
                        # Update the sheet name to the new one
                        update['sheet_name'] = new_sheet['title']
                        break
                modified_updates.append(update)
            
            # Replace updates with modified updates
            mapping_result['updates'] = modified_updates
        
        # Apply the mapping to the sheet
        result = apply_mapping_result(sheet_url, mapping_result)
        
        return jsonify({
            'success': True,
            'message': 'Data mapped and sheet updated successfully',
            'mapping': mapping_result,
            'update_result': result,
            'spreadsheet_id': extract_spreadsheet_id(sheet_url)
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
    if 'pdf_filepath' not in session:
        return jsonify({'error': 'No PDF data found. Please upload a PDF first.'}), 400
    
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400
    
    sheet_url = request.form.get('sheet_url')
    if not sheet_url:
        return jsonify({'error': 'Google Sheet URL is required'}), 400
        
    try:
        # Load PDF data from cached file instead of re-processing
        pdf_data = load_pdf_data()
        
        # Get sheet structure
        sheet_structure = get_sheet_structure(sheet_url)
        
        # Process the query (implement this function in a query_processor.py file)
        # For now, let's just return a placeholder
        result = {
            'query': query,
            'response': f"Query processing is not implemented yet. You asked: {query}"
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 