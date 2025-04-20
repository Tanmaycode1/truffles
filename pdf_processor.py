import PyPDF2
import pdfplumber
import pandas as pd
import re
import tabula
import json
import os
import tempfile
import shutil
from PIL import Image
import io
import base64
import requests
# import fitz  # PyMuPDF - Removing this dependency
from dotenv import load_dotenv
import logging
import google.auth
from google.oauth2 import service_account
from googleapiclient.discovery import build
import openai  # Adding openai explicitly
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib  # For generating unique file names

# Note: pdf2image requires system dependencies:
# - On macOS: brew install poppler
# - On Ubuntu/Debian: apt-get install poppler-utils
# - On Windows: Install poppler from http://blog.alivate.com.au/poppler-windows/
try:
    from pdf2image import convert_from_path, convert_from_bytes
    pdf2image_available = True
except ImportError:
    pdf2image_available = False
    logging.warning("pdf2image not installed. Using PyPDF2 for basic PDF extraction instead.")

# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# Path for storing extracted PDF data
def get_pdf_data_dir():
    """
    Get or create directory for storing extracted PDF data
    
    Returns:
        str: Path to the PDF data directory
    """
    # Create in user's home directory to persist between sessions
    pdf_data_dir = os.path.join(os.path.expanduser("~"), '.truffles', 'pdf_data')
    os.makedirs(pdf_data_dir, exist_ok=True)
    return pdf_data_dir

def save_pdf_data(pdf_path, financial_data):
    """
    Save extracted PDF data to a file for later use
    
    Args:
        pdf_path (str): Path to the original PDF file
        financial_data (dict): Extracted financial data
        
    Returns:
        str: Path to the saved data file
    """
    try:
        # Create a unique filename based on the PDF path
        pdf_basename = os.path.basename(pdf_path)
        pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:10]
        output_filename = f"{pdf_basename}_{pdf_hash}.json"
        
        # Save to the PDF data directory
        data_dir = get_pdf_data_dir()
        output_path = os.path.join(data_dir, output_filename)
        
        with open(output_path, 'w') as f:
            json.dump(financial_data, f, indent=2)
            
        logger.info(f"Saved extracted PDF data to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving PDF data: {e}")
        return None

def load_pdf_data(pdf_path=None, data_path=None):
    """
    Load previously extracted PDF data
    
    Args:
        pdf_path (str, optional): Path to the original PDF file
        data_path (str, optional): Direct path to the saved data file
        
    Returns:
        dict: Extracted financial data or None if not found
    """
    try:
        # If data_path is provided, use it directly
        if data_path and os.path.exists(data_path):
            with open(data_path, 'r') as f:
                logger.info(f"Loading PDF data from {data_path}")
                return json.load(f)
        
        # If pdf_path is provided, try to find the matching data file
        if pdf_path:
            pdf_basename = os.path.basename(pdf_path)
            pdf_hash = hashlib.md5(pdf_path.encode()).hexdigest()[:10]
            expected_filename = f"{pdf_basename}_{pdf_hash}.json"
            
            data_dir = get_pdf_data_dir()
            expected_path = os.path.join(data_dir, expected_filename)
            
            if os.path.exists(expected_path):
                with open(expected_path, 'r') as f:
                    logger.info(f"Loading PDF data from {expected_path}")
                    return json.load(f)
            else:
                logger.warning(f"No saved data found for {pdf_path}")
        
        # If no matching file is found
        return None
    except Exception as e:
        logger.error(f"Error loading PDF data: {e}")
        return None

# Create temporary directory for API responses
def get_temp_response_dir():
    """
    Get or create temporary directory for storing API responses
    
    Returns:
        str: Path to the temporary directory
    """
    temp_dir = os.path.join(tempfile.gettempdir(), 'truffles_responses')
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def save_api_response(response, page_num, response_type="vision"):
    """
    Save API response to temporary file for debugging and caching
    
    Args:
        response: The API response to save
        page_num (int): Page number
        response_type (str): Type of response (vision, kvp, etc.)
        
    Returns:
        str: Path to the saved response file
    """
    try:
        temp_dir = get_temp_response_dir()
        response_file = os.path.join(temp_dir, f"openai_{response_type}_page_{page_num}.json")
        
        with open(response_file, 'w') as f:
            if isinstance(response, dict):
                json.dump(response, f, indent=2)
            else:
                # For string responses
                f.write(str(response))
                
        logger.info(f"Saved {response_type} API response for page {page_num} to {response_file}")
        return response_file
    except Exception as e:
        logger.error(f"Error saving API response: {e}")
        return None

def load_api_response(page_num, response_type="vision"):
    """
    Load API response from temporary file if it exists
    
    Args:
        page_num (int): Page number
        response_type (str): Type of response (vision, kvp, etc.)
        
    Returns:
        dict or None: The loaded response or None if not found
    """
    try:
        temp_dir = get_temp_response_dir()
        response_file = os.path.join(temp_dir, f"openai_{response_type}_page_{page_num}.json")
        
        if os.path.exists(response_file):
            with open(response_file, 'r') as f:
                logger.info(f"Loading cached {response_type} API response for page {page_num}")
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading API response: {e}")
        return None

def cleanup_temp_responses():
    """
    Clean up temporary response files
    
    Returns:
        bool: Success status
    """
    try:
        temp_dir = get_temp_response_dir()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary response directory: {temp_dir}")
        return True
    except Exception as e:
        logger.error(f"Error cleaning up temporary responses: {e}")
        return False

# Google Sheets API integration

def get_sheets_service():
    """
    Get Google Sheets API service using service account credentials
    
    Returns:
        service: Google Sheets API service
    """
    try:
        # Get path to service account credentials file
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'credentials.json')
        
        if not os.path.exists(credentials_path):
            logger.error(f"Service account credentials file not found at {credentials_path}")
            return None
        
        # First validate the JSON file
        with open(credentials_path, 'r') as f:
            cred_content = f.read()
            try:
                # Try parsing the JSON
                json.loads(cred_content)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in service account file: {str(e)}")
                
                # Fix common JSON issues that might cause parsing errors
                if "\\n" in cred_content:
                    logger.info("Attempting to fix escaped newlines in JSON...")
                    cred_content = cred_content.replace("\\n", "\\\\n")
                    
                    # Write fixed content back to file
                    try:
                        with open(credentials_path, 'w') as fix_file:
                            fix_file.write(cred_content)
                        logger.info("Fixed credentials file and rewrote it")
                    except Exception as write_err:
                        logger.error(f"Could not write fixed credentials: {str(write_err)}")
                        return None
                else:
                    return None
        
        # Load credentials
        try:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            # Build the service
            service = build('sheets', 'v4', credentials=credentials)
            
            return service
        
        except Exception as cred_error:
            logger.error(f"Error initializing credentials: {str(cred_error)}")
            return None
    
    except Exception as e:
        logger.error(f"Error getting Google Sheets service: {str(e)}")
        return None

def get_sheet_names(spreadsheet_id):
    """
    Get all sheet names in a Google Sheets spreadsheet
    
    Args:
        spreadsheet_id: ID of the Google Sheets spreadsheet
        
    Returns:
        list: Sheet names
    """
    try:
        service = get_sheets_service()
        if not service:
            return []
        
        # Get spreadsheet metadata
        spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        
        # Extract sheet names
        sheets = spreadsheet.get('sheets', [])
        sheet_names = [sheet.get('properties', {}).get('title') for sheet in sheets]
        
        return sheet_names
    
    except Exception as e:
        logger.error(f"Error getting sheet names: {str(e)}")
        return []

def get_sheet_data(spreadsheet_id, sheet_name):
    """
    Get data from a specific sheet in a Google Sheets spreadsheet
    
    Args:
        spreadsheet_id: ID of the Google Sheets spreadsheet
        sheet_name: Name of the sheet to retrieve
        
    Returns:
        list: 2D array of sheet data
    """
    try:
        service = get_sheets_service()
        if not service:
            logger.error("Failed to get Google Sheets service")
            return None
        
        # Get the sheet data
        range_name = f"'{sheet_name}'!A1:Z1000"  # Adjust range as needed
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueRenderOption='UNFORMATTED_VALUE'
        ).execute()
        
        values = result.get('values', [])
        if not values:
            logger.info(f"No data found in sheet '{sheet_name}'")
            return []
        
        return values
        
    except Exception as e:
        logger.error(f"Error getting sheet data: {str(e)}")
        return None

def update_sheet_data(spreadsheet_id, sheet_name, values):
    """
    Update data in a specific sheet in Google Sheets
    
    Args:
        spreadsheet_id: ID of the Google Sheets spreadsheet
        sheet_name: Name of the sheet to update
        values: 2D array of values to write
        
    Returns:
        bool: Success status
    """
    try:
        service = get_sheets_service()
        if not service:
            return False
        
        # Prepare the update
        body = {
            'values': values
        }
        
        # Execute the update
        result = service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=sheet_name,
            valueInputOption='USER_ENTERED',
            body=body
        ).execute()
        
        # Check success
        updated_cells = result.get('updatedCells')
        return updated_cells > 0
    
    except Exception as e:
        logger.error(f"Error updating sheet data for {sheet_name}: {str(e)}")
        return False

def get_sheet_matrices(spreadsheet_id):
    """
    Get all sheets from a Google Sheets spreadsheet as matrices
    
    Args:
        spreadsheet_id: ID of the Google Sheets spreadsheet
        
    Returns:
        dict: Mapping of sheet names to their data matrices
    """
    try:
        # Get all sheet names
        sheet_names = get_sheet_names(spreadsheet_id)
        if not sheet_names:
            logger.warning(f"No sheets found in spreadsheet {spreadsheet_id}")
            return {}
        
        # Get data for each sheet
        matrices = {}
        for sheet_name in sheet_names:
            data = get_sheet_data(spreadsheet_id, sheet_name)
            if data:
                matrices[sheet_name] = data
        
        return matrices
    
    except Exception as e:
        logger.error(f"Error getting sheet matrices: {str(e)}")
        return {}

def extract_financial_data_from_pdf(pdf_path, max_workers=3, api_key=None):
    """
    Extract financial data from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        max_workers (int, optional): Maximum number of worker threads for parallel processing
        api_key (str, optional): OpenAI API key
        
    Returns:
        dict: Extracted financial data
    """
    logger.info(f"Starting extraction from {pdf_path}")
    
    financial_data = {
        'pages': [],
        'tables': [],
        'key_value_pairs': [],
        'metadata': {},
        'sheets': {}
    }
    
    try:
        # Open PDF with PyPDF2 for basic metadata
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            financial_data['metadata']['page_count'] = total_pages
            
            # Initialize pages data structure
            all_page_data = [{'page_num': i, 'text': '', 'tables': []} for i in range(total_pages)]
            
            # Extract text from all pages first (faster than processing one by one)
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                all_page_data[page_num]['text'] = page.extract_text()
            
            # Process images and extract tables in parallel
            if pdf2image_available:
                # Process pages in batches
                page_batches = []
                batch_size = min(4, total_pages)  # Process 4 pages at a time max
                
                for i in range(0, total_pages, batch_size):
                    page_batches.append(list(range(i, min(i + batch_size, total_pages))))
                
                # Process each batch with threading
                for batch in page_batches:
                    process_page_batch(pdf_path, batch, all_page_data, financial_data, max_workers)
            else:
                # Fallback to sequential processing with text images
                for page_num in range(total_pages):
                    text = all_page_data[page_num]['text']
                    text_img = create_text_image(text)
                    if text_img:
                        tables = extract_tables_with_openai(text_img, page_num)
                        if tables:
                            all_page_data[page_num]['tables'].extend(tables)
                            financial_data['tables'].extend(tables)
            
            # After all page data is processed and we have text, use OpenAI to extract key-value pairs
            # Collect text blocks from each page
            text_blocks = []
            for page_data in all_page_data:
                if page_data['text'].strip():
                    text_blocks.append(page_data['text'])
            
            # Use OpenAI to extract key-value pairs if we have text content
            kv_pairs = []
            if text_blocks:
                try:
                    kv_pairs = extract_key_value_pairs_with_openai(text_blocks, api_key)
                    print(f"Extracted {len(kv_pairs)} key-value pairs using OpenAI")
                except Exception as e:
                    print(f"Error extracting key-value pairs with OpenAI: {e}")
                    print("Falling back to traditional extraction method")
                    # Fall back to traditional extraction
                    key_value_pairs = extract_key_value_pairs(all_text)
                    for category, kv_dict in key_value_pairs.items():
                        for key, value in kv_dict.items():
                            kv_pairs.append({"key": key, "value": value, "category": category})
            
            # Process each category and add to data dictionary
            # Also add date detection results to key-value pairs
            financial_data['key_value_pairs'] = kv_pairs
            
            # Add page data to our collection
            financial_data['pages'] = all_page_data
        
        # 3. Detect financial sheet types in the document
        financial_data['metadata']['document_type'] = detect_document_type(financial_data)
        financial_data['metadata']['statement_date'] = detect_statement_date(financial_data)
        
        # 4. Organize data by sheet types (balance sheet, income statement, etc.)
        organize_by_sheet_types(financial_data)
        
        return financial_data
    
    except Exception as e:
        logger.error(f"Error extracting financial data: {str(e)}")
        return financial_data

def process_page_batch(pdf_path, page_nums, all_page_data, financial_data, max_workers):
    """
    Process a batch of pages in parallel
    
    Args:
        pdf_path: Path to PDF
        page_nums: List of page numbers to process
        all_page_data: Data structure to store page data
        financial_data: Main financial data structure
        max_workers: Maximum number of workers
    """
    # Convert pages to images
    dpi = 200  # Adjust DPI for quality vs speed
    first_page = min(page_nums) + 1  # pdf2image uses 1-indexed pages
    last_page = max(page_nums) + 1
    images = convert_from_path(pdf_path, first_page=first_page, last_page=last_page, dpi=dpi)
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        futures = {}
        for i, img in enumerate(images):
            page_num = page_nums[i]
            futures[executor.submit(extract_tables_with_openai, img, page_num)] = page_num
        
        # Process results as they complete
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                tables = future.result()
                if tables:
                    all_page_data[page_num]['tables'].extend(tables)
                    financial_data['tables'].extend(tables)
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {str(e)}")

def create_text_image(text):
    """
    Create a basic image from text content using PIL
    
    Args:
        text: Text to convert to image
        
    Returns:
        PIL Image: Generated image
    """
    try:
        # Basic image creation
        width, height = 1000, max(1000, len(text) // 3)  # Simple heuristic for size
        image = Image.new('RGB', (width, height), color=(255, 255, 255))
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, or use default
        try:
            font = ImageFont.truetype("Arial", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # Add text to image
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)
        
        return image
    
    except Exception as e:
        logger.error(f"Error creating text image: {str(e)}")
        return None

# Updated extract_tables_with_openai function to handle different image sources
def extract_tables_with_openai(img, page_num):
    """
    Extract tables from an image using OpenAI
    
    Args:
        img: PIL Image object of the page
        page_num: Page number
        
    Returns:
        list: Extracted tables
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.warning("OpenAI API key not found. Cannot extract tables.")
        return []
    
    tables = []
    
    try:
        # Convert image to base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Call OpenAI to extract tables
        response = call_openai_vision(img_base64, openai_api_key)
        if not response:
            logger.error("No response from OpenAI Vision API")
            return []
        
        # Parse the response
        extracted_tables = parse_openai_response(response, page_num)
        if extracted_tables:
            tables.extend(extracted_tables)
            logger.info(f"Successfully extracted {len(extracted_tables)} tables from page {page_num}")
        else:
            logger.warning(f"No tables extracted from page {page_num}")
        
        # If tables were successfully extracted, format them as HTML
        for table in tables:
            table['html'] = convert_table_to_html(table)
    
    except Exception as e:
        logger.error(f"Error extracting tables with OpenAI on page {page_num}: {str(e)}")
    
    return tables

def call_openai_vision(image_base64, api_key):
    """
    Call OpenAI Vision API to extract structured data from an image
    
    Args:
        image_base64 (str): Base64-encoded image
        api_key (str): OpenAI API key
        
    Returns:
        dict: OpenAI API response
    """
    # Set up API key
    openai.api_key = api_key
    
    # Define the prompt to extract structured data
    prompt = """
You are an expert in extracting financial data from images.
Please extract all tables from this image of a financial document.
For each table:
1. Identify the table type (e.g., Balance Sheet, Income Statement, Cash Flow Statement)
2. Extract all headers (columns and rows)
3. Extract all data in the table
4. Format as proper structured data

Return the extracted data as valid JSON with this structure:
[
  {
    "type": "Balance Sheet",
    "headers": ["Assets", "2023", "2022"],
    "data": [
      ["Current Assets", "100,000", "90,000"],
      ["Fixed Assets", "200,000", "180,000"],
      ["Total Assets", "300,000", "270,000"]
    ]
  },
  ... additional tables ...
]
"""
    
    try:
        # Create the request
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Use vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            timeout=60  # Add timeout to prevent long-running requests
        )
        
        # Save the response to a temporary file
        save_api_response(response, 0, "vision")  # We don't know the page number here, set later
        
        return response
    
    except Exception as e:
        logger.error(f"Error calling OpenAI Vision API: {str(e)}")
        return None

def fix_json_string(json_str):
    """
    Attempt to fix common JSON issues
    
    Args:
        json_str: Potentially broken JSON string
        
    Returns:
        str: Fixed JSON string
    """
    try:
        # Check if it's already valid
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        # Apply fixes
        cleaned = json_str
        
        # Replace single quotes with double quotes (common API response issue)
        cleaned = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', cleaned)
        
        # Fix unquoted property names
        cleaned = re.sub(r'([{,])\s*([A-Za-z0-9_]+)\s*:', r'\1"\2":', cleaned)
        
        # Remove trailing commas in objects and arrays
        cleaned = re.sub(r',\s*}', '}', cleaned)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        
        # Remove newlines within strings
        cleaned = re.sub(r'"\s*\n\s*', '" ', cleaned)
        
        try:
            # Check if our fixes worked
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            # If still not valid, try a more aggressive approach
            # Extract just what looks like a JSON object or array
            json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
            match = re.search(json_pattern, cleaned)
            if match:
                extracted = match.group(0)
                try:
                    json.loads(extracted)
                    return extracted
                except:
                    pass
            
            # If nothing works, return the original
            return json_str

def parse_openai_response(response, page_num):
    """
    Parse OpenAI Vision API response to extract tables
    
    Args:
        response: OpenAI API response
        page_num: Page number
        
    Returns:
        list: Extracted tables
    """
    tables = []
    
    if not response or 'choices' not in response or not response['choices']:
        logger.warning(f"Empty or invalid response from OpenAI API for page {page_num}")
        return tables
    
    try:
        # In v0.27.8, the response structure is different
        content = response['choices'][0]['message']['content']
        logger.info(f"Response from OpenAI on page {page_num}: {content[:100]}...")
        
        # Save raw response for debugging
        save_api_response(content, page_num, "vision_content")
        
        # Try to parse the response as JSON
        try:
            # Look for a JSON block in the content
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            
            if json_match:
                json_str = json_match.group(1)
                table_list = json.loads(json_str)
            else:
                # Try to parse the whole content as JSON
                json_str = fix_json_string(content)
                table_list = json.loads(json_str)
            
            # If we get a single table object, convert to a list
            if isinstance(table_list, dict):
                table_list = [table_list]
            
            if not isinstance(table_list, list):
                logger.warning(f"Unexpected JSON format on page {page_num}: {type(table_list)}")
                return tables
            
            for idx, table_data in enumerate(table_list):
                # Skip if incomplete
                if 'headers' not in table_data or 'data' not in table_data:
                    logger.warning(f"Incomplete table data on page {page_num}, index {idx}: missing headers or data")
                    continue
                
                # Make sure type is extracted properly
                table_type = table_data.get('type', '').strip()
                if not table_type or table_type.lower() == 'unknown':
                    # Try to infer type from data
                    headers = table_data.get('headers', [])
                    data_rows = table_data.get('data', [])
                    if any('balance' in str(h).lower() for h in headers):
                        table_type = 'Balance Sheet'
                    elif any('income' in str(h).lower() for h in headers):
                        table_type = 'Income Statement'
                    elif any('cash' in str(h).lower() for h in headers):
                        table_type = 'Cash Flow Statement'
                    elif len(data_rows) > 0:
                        # Look at first column for clues
                        first_col = [row[0] if len(row) > 0 else '' for row in data_rows]
                        if any('asset' in str(item).lower() for item in first_col) and any('liabilit' in str(item).lower() for item in first_col):
                            table_type = 'Balance Sheet'
                        elif any('revenue' in str(item).lower() for item in first_col) or any('income' in str(item).lower() for item in first_col):
                            table_type = 'Income Statement'
                        elif any('cash' in str(item).lower() for item in first_col) and any('flow' in str(item).lower() for item in first_col):
                            table_type = 'Cash Flow Statement'
                    
                    if not table_type:
                        table_type = 'Financial Table'
                
                headers = table_data['headers']
                data = table_data['data']
                
                tables.append({
                    'page': page_num,
                    'type': table_type,
                    'headers': headers,
                    'data': data,
                    'source': f'openai_{idx}'
                })
                logger.info(f"Added table of type '{table_type}' from page {page_num}")
        
        except json.JSONDecodeError as json_err:
            logger.warning(f"Could not parse JSON from OpenAI response on page {page_num}: {str(json_err)}")
            logger.debug(f"Response content: {content}")
            
            # Try to parse as markdown table as a fallback
            markdown_tables = extract_markdown_tables(content)
            for idx, md_table in enumerate(markdown_tables):
                tables.append({
                    'page': page_num,
                    'type': 'Financial Table',
                    'headers': md_table['headers'],
                    'data': md_table['data'],
                    'source': f'markdown_{idx}'
                })
                logger.info(f"Added table from markdown parsing on page {page_num}")
    
    except Exception as e:
        logger.error(f"Error parsing OpenAI response: {str(e)}")
    
    logger.info(f"Extracted {len(tables)} tables from page {page_num}")
    return tables

def extract_markdown_tables(text):
    """
    Extract tables from markdown text
    
    Args:
        text: Markdown text possibly containing tables
        
    Returns:
        list: Extracted tables
    """
    tables = []
    
    # Split text into lines
    lines = text.split('\n')
    
    current_headers = []
    current_data = []
    in_table = False
    
    for line in lines:
        stripped = line.strip()
        
        # Check if line is part of a markdown table
        if stripped.startswith('|') and stripped.endswith('|'):
            # Extract cells
            cells = [cell.strip() for cell in stripped.strip('|').split('|')]
            
            if not in_table:
                # Start of a new table
                in_table = True
                current_headers = cells
                current_data = []
            elif all(re.match(r'^-+$', cell) for cell in cells if cell):
                # This is a separator row, skip it
                continue
            else:
                # This is a data row
                current_data.append(cells)
        else:
            # Not a table row
            if in_table and current_headers and current_data:
                # End of a table, save it
                tables.append({
                    'headers': current_headers,
                    'data': current_data
                })
                current_headers = []
                current_data = []
                in_table = False
    
    # Check if we have a table at the end
    if in_table and current_headers and current_data:
        tables.append({
            'headers': current_headers,
            'data': current_data
        })
    
    return tables

def convert_table_to_html(table):
    """
    Convert a table to HTML format
    
    Args:
        table: Table dictionary with headers and data
        
    Returns:
        str: HTML representation of the table
    """
    headers = table.get('headers', [])
    data = table.get('data', [])
    table_type = table.get('type', '')
    
    html = f'<table class="financial-table {table_type.lower().replace(" ", "-")}">\n'
    
    # Add caption if we have a type
    if table_type and table_type != 'unknown':
        html += f'  <caption>{table_type}</caption>\n'
    
    # Add headers
    html += '  <thead>\n    <tr>\n'
    for header in headers:
        html += f'      <th>{header}</th>\n'
    html += '    </tr>\n  </thead>\n'
    
    # Add data rows
    html += '  <tbody>\n'
    for row in data:
        html += '    <tr>\n'
        for i, cell in enumerate(row):
            # If first column, use th instead of td (row headers)
            if i == 0 and headers:
                html += f'      <th>{cell}</th>\n'
            else:
                html += f'      <td>{cell}</td>\n'
        html += '    </tr>\n'
    html += '  </tbody>\n'
    
    html += '</table>'
    return html

def detect_document_type(financial_data):
    """
    Detect the type of financial document based on extracted data
    
    Args:
        financial_data: Dictionary containing extracted data
        
    Returns:
        str: Detected document type
    """
    # Get all text from all pages
    all_text = ' '.join([page.get('text', '') for page in financial_data.get('pages', [])])
    all_text = all_text.lower()
    
    # Check for common document types
    if any(term in all_text for term in ['balance sheet', 'assets', 'liabilities', 'equity']):
        return 'balance_sheet'
    elif any(term in all_text for term in ['income statement', 'profit and loss', 'revenue', 'expenses']):
        return 'income_statement'
    elif any(term in all_text for term in ['cash flow', 'operating activities', 'investing activities']):
        return 'cash_flow'
    elif any(term in all_text for term in ['changes in equity', 'statement of equity']):
        return 'equity_statement'
    else:
        return 'financial_report'

def detect_statement_date(financial_data):
    """
    Detect the statement date from the extracted data
    
    Args:
        financial_data: Dictionary containing extracted data
        
    Returns:
        str: Detected statement date
    """
    # First check key-value pairs for date-related keys
    date_related_keys = ['date', 'as of', 'as at', 'period ended', 'year ended']
    
    for kv in financial_data.get('key_value_pairs', []):
        key = kv.get('key', '').lower()
        if any(date_key in key for date_key in date_related_keys):
            return kv.get('value')
    
    # If not found in key-value pairs, try regex on text
    all_text = ' '.join([page.get('text', '') for page in financial_data.get('pages', [])])
    
    # Date patterns
    patterns = [
        r'(?:as at|as of|for the year[s]? ended|for the period ended)\s+(\d{1,2}\s+[a-zA-Z]+\s+\d{4})',
        r'(?:as at|as of|for the year[s]? ended|for the period ended)\s+([a-zA-Z]+\s+\d{1,2},?\s+\d{4})',
        r'(?:as at|as of|for the year[s]? ended|for the period ended)\s+(\d{4}-\d{1,2}-\d{1,2})',
        r'(?:as at|as of|for the year[s]? ended|for the period ended)\s+(\d{1,2}/\d{1,2}/\d{4})',
        r'(\d{1,2}\s+[a-zA-Z]+\s+\d{4})',
        r'([a-zA-Z]+\s+\d{1,2},?\s+\d{4})',
        r'(\d{4}-\d{1,2}-\d{1,2})',
        r'(\d{1,2}/\d{1,2}/\d{4})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def organize_by_sheet_types(financial_data):
    """
    Organize extracted data by financial sheet types
    
    Args:
        financial_data: Dictionary to organize
    """
    # Sheet types to look for
    sheet_types = ['balance_sheet', 'income_statement', 'cash_flow', 'equity_statement']
    
    # Initialize sheets
    for sheet_type in sheet_types:
        financial_data['sheets'][sheet_type] = {
            'tables': [],
            'key_value_pairs': [],
            'pages': []
        }
    
    # Categorize tables by their type
    for table in financial_data.get('tables', []):
        table_type = table.get('type', '').lower()
        
        if 'balance' in table_type:
            financial_data['sheets']['balance_sheet']['tables'].append(table)
        elif 'income' in table_type or 'profit' in table_type:
            financial_data['sheets']['income_statement']['tables'].append(table)
        elif 'cash flow' in table_type:
            financial_data['sheets']['cash_flow']['tables'].append(table)
        elif 'equity' in table_type:
            financial_data['sheets']['equity_statement']['tables'].append(table)
    
    # Add page references
    for sheet_type in sheet_types:
        tables = financial_data['sheets'][sheet_type]['tables']
        if tables:
            financial_data['sheets'][sheet_type]['pages'] = list(set(table['page'] for table in tables))

def extract_key_value_pairs(text):
    """
    Extract key-value pairs from text
    
    Args:
        text: Text to extract from
        
    Returns:
        list: Extracted key-value pairs
    """
    key_value_pairs = []
    
    if not text:
        return key_value_pairs
    
    # Patterns to match key-value pairs
    patterns = [
        # "Key: Value" format
        r'([A-Za-z][A-Za-z\s\',\(\)&-]+):\s*([\d,\.\-\(\)$€£\w\s]+)',
        
        # Key-Value format with space separation (value is numeric)
        r'([A-Za-z][A-Za-z\s\',\(\)&-]+)\s+([\d,\.\-\(\)$€£]+)$',
        
        # Financial statement line items and values
        r'([A-Za-z][A-Za-z\s\',\(\)&-]+)\s+([\d,\.\-\(\)$€£]+)'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            key, value = match
            
            # Clean and normalize
            key = key.strip()
            value = value.strip()
            
            # Skip if too short or likely a false positive
            if len(key) < 3 or len(value) < 1:
                continue
            
            # Handle numeric values
            numeric_value = None
            if re.match(r'^[\d,\.\-\(\)$€£]+$', value):
                # Clean value for numeric conversion
                clean_value = value.replace(',', '').replace('$', '').replace('€', '').replace('£', '')
                
                # Handle negative numbers in parentheses
                if '(' in clean_value and ')' in clean_value:
                    clean_value = clean_value.replace('(', '-').replace(')', '')
                
                try:
                    numeric_value = float(clean_value)
                except ValueError:
                    pass
            
            # Add to results if not a duplicate
            if not any(kv['key'] == key for kv in key_value_pairs):
                key_value_pairs.append({
                    'key': key,
                    'value': value,
                    'numeric_value': numeric_value,
                    'confidence': 0.8
                })
    
    return key_value_pairs

def fill_matrix_with_financial_data(matrix, financial_data, user_query=None):
    """
    Use OpenAI to intelligently fill a matrix with financial data while preserving structure
    
    Args:
        matrix: Google Sheets matrix to fill
        financial_data: Extracted financial data
        user_query: Optional user query to focus the filling
        
    Returns:
        list: Filled matrix with same structure as original
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OpenAI API key not found. Cannot fill matrix.")
        return matrix
    
    try:
        # Set the API key
        openai.api_key = openai_api_key
        
        # Make a deep copy of the original matrix to avoid modifying it
        result_matrix = [row[:] for row in matrix]
        
        # Convert data to JSON for API call
        matrix_json = json.dumps(matrix)
        
        # Get matrix dimensions for verification later
        original_rows = len(matrix)
        original_cols = len(matrix[0]) if original_rows > 0 else 0
        
        # Don't send the entire financial data - it can be too large
        # Extract key information only
        simplified_financial_data = {
            "key_value_pairs": financial_data.get("key_value_pairs", []),
            "metadata": financial_data.get("metadata", {}),
            "tables": []
        }
        
        # Include only essential table data with limited rows
        for table in financial_data.get("tables", []):
            table_copy = {
                "type": table.get("type", "Unknown"),
                "headers": table.get("headers", []),
                "data": table.get("data", [])[:10]  # Limit number of rows
            }
            simplified_financial_data["tables"].append(table_copy)
        
        financial_data_json = json.dumps(simplified_financial_data)
        
        # Create system prompt
        system_prompt = """You are a financial data processing assistant specialized in filling spreadsheet matrices.
Your task is to fill EMPTY cells in a matrix with relevant financial data from the extracted data.
The filled matrix MUST have EXACTLY the same structure, dimensions, and organization as the original matrix.
You MUST preserve ALL existing values and formatting in the original matrix.
You MUST return ONLY a valid JSON array of arrays representing the filled matrix."""
        
        # Create a prompt based on user query
        if user_query:
            prompt = f"""
I need you to fill EMPTY cells in this Google Sheets matrix with financial data from a PDF. 

IMPORTANT: You MUST preserve the EXACT structure and organization of the original matrix.

USER QUERY: {user_query}

Original matrix (DO NOT CHANGE STRUCTURE OR EXISTING VALUES):
{matrix_json}

Financial data extracted from PDF:
{financial_data_json}

INSTRUCTIONS:
1. Focus on addressing the user's query specifically
2. ONLY fill in EMPTY cells (cells with "", null, undefined) or cells with placeholders like "N/A" or "TBD"
3. NEVER change the structure, dimensions, or organization of the matrix
4. NEVER modify existing data values - only add data to empty cells
5. Use the same formatting as surrounding cells for consistency
6. If exact data isn't available, leave the cell empty rather than making something up
7. Handle different naming conventions between the sheet and the data

The output MUST be a valid JSON array of arrays with EXACTLY the same dimensions as the input matrix: {original_rows}×{original_cols}.
EVERY row must have EXACTLY {original_cols} columns.
"""
        else:
            prompt = f"""
I need you to fill EMPTY cells in this Google Sheets matrix with financial data from a PDF.

IMPORTANT: You MUST preserve the EXACT structure and organization of the original matrix.

Original matrix (DO NOT CHANGE STRUCTURE OR EXISTING VALUES):
{matrix_json}

Financial data extracted from PDF:
{financial_data_json}

INSTRUCTIONS:
1. ONLY fill in EMPTY cells (cells with "", null, undefined) or cells with placeholders like "N/A" or "TBD"
2. NEVER change the structure, dimensions, or organization of the matrix
3. NEVER modify existing data values - only add data to empty cells
4. Use the same formatting as surrounding cells for consistency
5. Match column and row headers in the matrix with equivalent data in the financial data
6. Handle different naming conventions between the sheet and the data
7. Focus on financial values, dates, and key metrics

The output MUST be a valid JSON array of arrays with EXACTLY the same dimensions as the input matrix: {original_rows}×{original_cols}.
EVERY row must have EXACTLY {original_cols} columns.
"""
        
        # Call OpenAI API
        logger.info("Calling OpenAI to fill matrix with financial data")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=8000,  # Increased max tokens for full context
            response_format={"type": "json_object"}  # Force JSON response format
        )
        
        # Extract the response content
        content = response.choices[0].message['content'].strip()
        logger.info(f"Received response from OpenAI (length: {len(content)})")
        
        # Parse the JSON response
        try:
            # Parse the content
            parsed_content = json.loads(content)
            
            # Check if the parsed content is an array (the matrix) or a wrapper object
            if isinstance(parsed_content, list):
                filled_matrix = parsed_content
            elif isinstance(parsed_content, dict) and "matrix" in parsed_content:
                filled_matrix = parsed_content["matrix"]
            else:
                # Look for any array property in the response that could be the matrix
                for key, value in parsed_content.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
                        filled_matrix = value
                        break
                else:
                    logger.error("Could not find matrix in response")
                    return result_matrix  # Return the copy of the original
            
            # Verify dimensions match original matrix
            if len(filled_matrix) != original_rows:
                logger.error(f"Filled matrix has {len(filled_matrix)} rows, but original has {original_rows} rows")
                return result_matrix
                
            # Check each row's length
            for i, row in enumerate(filled_matrix):
                if len(row) != original_cols:
                    logger.error(f"Row {i} in filled matrix has {len(row)} columns, but original has {original_cols} columns")
                    return result_matrix
            
            # Verify that existing data hasn't been changed
            for i in range(original_rows):
                for j in range(original_cols):
                    original_value = matrix[i][j]
                    filled_value = filled_matrix[i][j]
                    
                    # Check if original cell had a value and if it's been preserved
                    if original_value is not None and original_value != "" and original_value != "N/A" and original_value != "TBD":
                        # For numeric values, compare them approximately
                        if isinstance(original_value, (int, float)) and isinstance(filled_value, (int, float)):
                            if abs(original_value - filled_value) > 0.01:  # Allow small rounding differences
                                logger.warning(f"Value at [{i}][{j}] changed from {original_value} to {filled_value}. Keeping original.")
                                filled_matrix[i][j] = original_value
                        # For strings and other types, compare them exactly
                        elif str(original_value) != str(filled_value):
                            logger.warning(f"Value at [{i}][{j}] changed from {original_value} to {filled_value}. Keeping original.")
                            filled_matrix[i][j] = original_value
            
            logger.info("Successfully filled matrix with financial data while preserving structure")
            return filled_matrix
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.debug(f"Raw response: {content[:500]}...")  # Debug first 500 chars
            return result_matrix
            
    except Exception as e:
        logger.error(f"Error filling matrix with financial data: {str(e)}")
        # Return a copy of the original matrix if we encounter any error
        return [row[:] for row in matrix]

def map_data_to_sheets(financial_data):
    """
    Map financial data to appropriate sheets based on type
    
    Args:
        financial_data: Dictionary containing extracted financial data
        
    Returns:
        dict: Mapped data ready for sheets update
    """
    sheet_mappings = {
        'balance_sheet': {},
        'income_statement': {},
        'cash_flow': {},
        'equity_statement': {}
    }
    
    try:
        # Process tables based on their type
        for table in financial_data.get('tables', []):
            table_type = table.get('type', '').lower()
            
            # Determine which sheet to map to
            target_sheet = None
            if 'balance' in table_type:
                target_sheet = 'balance_sheet'
            elif 'income' in table_type or 'profit' in table_type or 'revenue' in table_type:
                target_sheet = 'income_statement'
            elif 'cash flow' in table_type or 'cash flows' in table_type:
                target_sheet = 'cash_flow'
            elif 'equity' in table_type:
                target_sheet = 'equity_statement'
            
            # If we've identified a target sheet, add the data
            if target_sheet:
                headers = table.get('headers', [])
                data = table.get('data', [])
                
                # Convert to matrix format
                matrix = [headers] + data
                
                # Store in the appropriate sheet mapping
                sheet_mappings[target_sheet] = matrix
        
        # If any sheet is missing data, check key-value pairs
        for sheet_type in sheet_mappings:
            if not sheet_mappings[sheet_type]:
                # Extract relevant key-value pairs for this sheet type
                relevant_kvs = []
                for kv in financial_data.get('key_value_pairs', []):
                    key = kv.get('key', '').lower()
                    
                    # Check if this key belongs in this sheet type
                    if sheet_type == 'balance_sheet' and any(term in key for term in ['asset', 'liabilit', 'equity', 'total']):
                        relevant_kvs.append(kv)
                    elif sheet_type == 'income_statement' and any(term in key for term in ['revenue', 'expense', 'income', 'profit', 'loss']):
                        relevant_kvs.append(kv)
                    elif sheet_type == 'cash_flow' and any(term in key for term in ['cash', 'operating', 'investing', 'financing']):
                        relevant_kvs.append(kv)
                    elif sheet_type == 'equity_statement' and any(term in key for term in ['equity', 'capital', 'retained', 'reserve']):
                        relevant_kvs.append(kv)
                
                if relevant_kvs:
                    # Convert to a simple 2-column matrix
                    matrix = [['Item', 'Value']]
                    for kv in relevant_kvs:
                        matrix.append([kv.get('key', ''), kv.get('value', '')])
                    
                    sheet_mappings[sheet_type] = matrix
        
        return sheet_mappings
    
    except Exception as e:
        logger.error(f"Error mapping data to sheets: {str(e)}")
        return sheet_mappings

def update_sheets_with_financial_data(spreadsheet_id, financial_data, user_query=None, sheet_names=None):
    """
    Update Google Sheets with extracted financial data based on user query
    
    Args:
        spreadsheet_id: ID of the Google Sheets spreadsheet
        financial_data: Extracted financial data
        user_query: Optional user query to focus the update
        sheet_names: Optional list of sheet names to update (if None, updates all)
        
    Returns:
        dict: Results of the update operation
    """
    results = {
        'success': False,
        'sheets_updated': [],
        'errors': []
    }
    
    try:
        # Get current sheet names from the spreadsheet
        available_sheets = get_sheet_names(spreadsheet_id)
        if not available_sheets:
            logger.error("Could not retrieve sheet names")
            results['errors'].append("Failed to get sheet names from spreadsheet")
            return results
        
        logger.info(f"Available sheets: {available_sheets}")
        
        # Map extracted data to sheet structures
        sheet_data_mappings = map_data_to_sheets(financial_data)
        
        # Filter sheet names if specified
        if sheet_names:
            target_sheets = [name for name in sheet_names if name in available_sheets]
        else:
            # Use default mappings
            sheet_mapping = {
                'balance_sheet': ['Balance Sheet', 'BalanceSheet', 'Balance'],
                'income_statement': ['Income Statement', 'IncomeStatement', 'Income', 'P&L', 'Profit and Loss'],
                'cash_flow': ['Cash Flow', 'CashFlow', 'Cash'],
                'equity_statement': ['Equity', 'EquityStatement', 'Statement of Equity']
            }
            
            # Match available sheets to our mappings
            target_sheets = []
            for sheet_type, possible_names in sheet_mapping.items():
                for name in possible_names:
                    matches = [s for s in available_sheets if name.lower() in s.lower()]
                    if matches:
                        target_sheets.extend(matches)
                        break
            
            # If no matches, use all sheets
            if not target_sheets:
                target_sheets = available_sheets
                logger.info("No specific sheets matched, using all available sheets")
        
        logger.info(f"Target sheets for update: {target_sheets}")
        
        # Get matrices from the sheets to update
        matrices = {}
        for sheet_name in target_sheets:
            data = get_sheet_data(spreadsheet_id, sheet_name)
            if data:
                matrices[sheet_name] = data
                logger.info(f"Retrieved matrix for {sheet_name}: {len(data)}x{len(data[0]) if data else 0}")
            else:
                logger.warning(f"Could not retrieve data for sheet: {sheet_name}")
        
        if not matrices:
            logger.error("No sheet data retrieved")
            results['errors'].append("Failed to get sheet data")
            return results
        
        # Update each sheet
        for sheet_name, matrix in matrices.items():
            try:
                logger.info(f"Processing sheet: {sheet_name}")
                
                # Determine the type of this sheet
                sheet_type = None
                sheet_name_lower = sheet_name.lower()
                if any(term in sheet_name_lower for term in ['balance']):
                    sheet_type = 'balance_sheet'
                elif any(term in sheet_name_lower for term in ['income', 'profit', 'loss', 'p&l']):
                    sheet_type = 'income_statement'
                elif any(term in sheet_name_lower for term in ['cash']):
                    sheet_type = 'cash_flow'
                elif any(term in sheet_name_lower for term in ['equity']):
                    sheet_type = 'equity_statement'
                
                mapped_data = None
                if sheet_type and sheet_type in sheet_data_mappings:
                    mapped_data = sheet_data_mappings[sheet_type]
                
                # Fill the matrix with financial data
                if mapped_data and len(mapped_data) > 0:
                    # Use specific data for this sheet type if available
                    logger.info(f"Using mapped data for {sheet_name} ({sheet_type})")
                    filled_matrix = integrate_data_into_matrix(matrix, mapped_data)
                else:
                    # Otherwise use the general approach with all data
                    logger.info(f"Using general data filling for {sheet_name}")
                    filled_matrix = fill_matrix_with_financial_data(matrix, financial_data, user_query)
                
                # Check if matrix changed
                if filled_matrix == matrix:
                    logger.info(f"No changes to sheet: {sheet_name}")
                    continue
                
                # Update the sheet
                success = update_sheet_data(spreadsheet_id, sheet_name, filled_matrix)
                
                if success:
                    logger.info(f"Successfully updated sheet: {sheet_name}")
                    results['sheets_updated'].append(sheet_name)
                else:
                    logger.error(f"Failed to update sheet: {sheet_name}")
                    results['errors'].append(f"Failed to update sheet: {sheet_name}")
            
            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
                results['errors'].append(f"Error processing sheet {sheet_name}: {str(e)}")
        
        # Set overall success
        results['success'] = len(results['sheets_updated']) > 0 and len(results['errors']) == 0
        
        return results
    
    except Exception as e:
        logger.error(f"Error updating sheets with financial data: {str(e)}")
        results['errors'].append(str(e))
        return results

def integrate_data_into_matrix(matrix, data_matrix):
    """
    Intelligently integrate extracted data into an existing matrix
    
    Args:
        matrix: Existing Google Sheets matrix
        data_matrix: Extracted data in matrix format
        
    Returns:
        list: Updated matrix
    """
    # Make a deep copy of the original matrix to avoid modifying it
    result_matrix = [row[:] for row in matrix]
    
    try:
        if not data_matrix or not matrix:
            return result_matrix
        
        # Extract headers from both matrices
        matrix_headers = matrix[0] if matrix else []
        data_headers = data_matrix[0] if data_matrix else []
        
        # Match headers between the matrices
        header_mapping = {}
        for i, data_header in enumerate(data_headers):
            data_header_lower = str(data_header).lower().strip()
            
            best_match = None
            best_match_score = 0
            
            for j, matrix_header in enumerate(matrix_headers):
                matrix_header_lower = str(matrix_header).lower().strip()
                
                # Calculate similarity score
                if data_header_lower == matrix_header_lower:
                    # Exact match
                    best_match = j
                    best_match_score = 100
                    break
                elif data_header_lower in matrix_header_lower or matrix_header_lower in data_header_lower:
                    # Partial match
                    score = 80
                    if score > best_match_score:
                        best_match = j
                        best_match_score = score
                elif any(year in data_header for year in ['2020', '2021', '2022', '2023', '2024']):
                    # Year columns
                    if any(year in matrix_header for year in ['2020', '2021', '2022', '2023', '2024']):
                        score = 90
                        if score > best_match_score:
                            best_match = j
                            best_match_score = score
            
            if best_match is not None:
                header_mapping[i] = best_match
        
        # Now match rows by looking at the first column of each matrix
        for i in range(1, len(data_matrix)):  # Skip header row
            data_row = data_matrix[i]
            if not data_row:
                continue
            
            # Get the row label (first column)
            data_label = str(data_row[0]).lower().strip()
            
            # Find matching row in the matrix
            best_match = None
            best_match_score = 0
            
            for j in range(1, len(matrix)):  # Skip header row
                matrix_row = matrix[j]
                if not matrix_row:
                    continue
                
                matrix_label = str(matrix_row[0]).lower().strip() if matrix_row else ""
                
                # Calculate similarity score
                if data_label == matrix_label:
                    # Exact match
                    best_match = j
                    best_match_score = 100
                    break
                elif data_label in matrix_label or matrix_label in data_label:
                    # Partial match
                    score = 80
                    if score > best_match_score:
                        best_match = j
                        best_match_score = score
                elif any(keyword in data_label for keyword in ['total', 'asset', 'liability', 'revenue', 'expense']):
                    # Important keywords
                    if any(keyword in matrix_label for keyword in ['total', 'asset', 'liability', 'revenue', 'expense']):
                        score = 70
                        if score > best_match_score:
                            best_match = j
                            best_match_score = score
            
            # If we found a matching row, update it
            if best_match is not None:
                # Update each column according to our header mapping
                for data_col, matrix_col in header_mapping.items():
                    if data_col < len(data_row) and matrix_col < len(result_matrix[best_match]):
                        data_value = data_row[data_col]
                        if data_value:  # Only update if we have a value
                            result_matrix[best_match][matrix_col] = data_value
        
        return result_matrix
    
    except Exception as e:
        logger.error(f"Error integrating data into matrix: {str(e)}")
        return result_matrix

def process_financial_pdf_to_sheets(pdf_path, spreadsheet_id, user_query=None, sheet_names=None, reuse_data=True):
    """
    Complete workflow to process a financial PDF and update Google Sheets
    
    Args:
        pdf_path: Path to the PDF file
        spreadsheet_id: ID of the Google Sheets spreadsheet
        user_query: Optional user query to focus the update
        sheet_names: Optional list of sheet names to update
        reuse_data: Whether to reuse previously extracted data if available
        
    Returns:
        dict: Results of the operation
    """
    results = {
        'success': False,
        'pdf_processed': False,
        'sheets_updated': [],
        'errors': []
    }
    
    try:
        logger.info(f"Starting financial PDF processing: {pdf_path}")
        
        # Check if we have previously extracted data for this PDF
        financial_data = None
        if reuse_data:
            financial_data = load_pdf_data(pdf_path)
            if financial_data:
                logger.info(f"Using previously extracted data for {pdf_path}")
                results['pdf_processed'] = True
        
        # Extract data from PDF if we don't have it already
        if not financial_data:
            financial_data = extract_financial_data_from_pdf(pdf_path)
            
            if not financial_data or 'pages' not in financial_data or not financial_data['pages']:
                logger.error("Failed to extract data from PDF")
                results['errors'].append("Failed to extract data from PDF")
                cleanup_temp_responses()  # Clean up temp files
                return results
            
            results['pdf_processed'] = True
            logger.info(f"Successfully extracted data from PDF: {len(financial_data['pages'])} pages, {len(financial_data['tables'])} tables")
            
            # Save the extracted data for future use
            data_path = save_pdf_data(pdf_path, financial_data)
            if data_path:
                results['data_saved'] = data_path
        
        # 2. Update Google Sheets with extracted data
        if spreadsheet_id:
            sheets_results = update_sheets_with_financial_data(
                spreadsheet_id,
                financial_data,
                user_query,
                sheet_names
            )
            
            # Merge results
            results['success'] = sheets_results['success']
            results['sheets_updated'] = sheets_results['sheets_updated']
            results['errors'].extend(sheets_results['errors'])
        else:
            logger.warning("No spreadsheet ID provided, skipping Google Sheets update")
        
        # 3. Clean up temporary files
        cleanup_temp_responses()
        
        return results
    
    except Exception as e:
        logger.error(f"Error in processing financial PDF to sheets: {str(e)}")
        results['errors'].append(str(e))
        cleanup_temp_responses()  # Clean up even on error
        return results

def extract_key_value_pairs_with_openai(text_blocks, api_key=None):
    """
    Use OpenAI to extract key-value pairs from text blocks
    
    Args:
        text_blocks (list): List of text blocks to process
        api_key (str, optional): OpenAI API key
        
    Returns:
        list: List of key-value pairs
    """
    import openai
    
    # Check for API key
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
    
    openai.api_key = api_key
    
    # Generate a hash of the text content to use as a cache key
    text_content = "\n---\n".join(text_blocks)
    content_hash = hashlib.md5(text_content.encode()).hexdigest()
    
    # Check for cached response
    cached_response = load_api_response(content_hash, "kvp")
    if cached_response:
        return cached_response
    
    # Prepare the prompt for key-value extraction
    system_prompt = "You are a financial document analysis assistant. Extract key-value pairs from financial documents accurately."
    
    prompt = f"""
Extract all key-value pairs from the following financial document text. 
Format as a JSON array of objects with 'key' and 'value' properties.
Focus on important financial information like:
- Document type (income statement, balance sheet, etc.)
- Company/entity name
- Dates (statement date, fiscal year, etc.)
- Time periods
- Totals and subtotals
- Financial metrics and their values
- Accounting categories and their values

TEXT:
{text_content}

Return ONLY valid JSON array in this format:
[
  {{"key": "document_type", "value": "Income Statement"}},
  {{"key": "company_name", "value": "Example Corp"}},
  {{"key": "period_ending", "value": "December 31, 2023"}},
  ...
]
"""

    try:
        # Call OpenAI API for key-value extraction using ChatCompletion API with powerful model
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Using model with larger context window
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent, deterministic outputs
            max_tokens=2000
        )
        
        # Get the response content
        content = response.choices[0].message['content'].strip()
        
        # Save the response to a temporary file
        save_api_response(content, content_hash, "kvp")
        
        # Extract JSON from the response
        import re
        json_match = re.search(r'(\[\s*\{.*\}\s*\])', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            
            try:
                # Parse the JSON response
                import json
                key_value_pairs = json.loads(json_str)
                save_api_response(key_value_pairs, content_hash, "kvp_parsed")
                return key_value_pairs
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                # Try to extract and fix JSON as a fallback
                return extract_json_fallback(content)
        else:
            print("No valid JSON found in response")
            return extract_json_fallback(content)
            
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        # Fall back to traditional extraction if AI fails
        return []

def extract_json_fallback(content):
    """Fallback method to extract valid JSON objects from text"""
    import re
    import json
    
    # Try to extract objects one by one
    results = []
    pattern = r'\{\s*"key"\s*:\s*"([^"]*)"\s*,\s*"value"\s*:\s*"([^"]*)"\s*\}'
    matches = re.findall(pattern, content)
    
    for key, value in matches:
        results.append({"key": key, "value": value})
    
    if results:
        return results
    
    # Last resort: try to create key-value pairs from lines with colons
    pairs = []
    lines = content.split('\n')
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().strip('"\'').lower().replace(' ', '_')
                value = parts[1].strip().strip('",\'')
                if key and value:
                    pairs.append({"key": key, "value": value})
    
    return pairs

# Add standalone function for extraction only
def extract_and_save_pdf_data(pdf_path, output_path=None):
    """
    Extract data from a PDF and save it to a file without mapping to sheets
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str, optional): Custom path to save the data
        
    Returns:
        dict: Result with paths and status
    """
    result = {
        'success': False,
        'pdf_processed': False,
        'data_path': None,
        'errors': []
    }
    
    try:
        logger.info(f"Extracting data from PDF: {pdf_path}")
        
        # Extract data from PDF
        financial_data = extract_financial_data_from_pdf(pdf_path)
        
        if not financial_data or 'pages' not in financial_data or not financial_data['pages']:
            logger.error("Failed to extract data from PDF")
            result['errors'].append("Failed to extract data from PDF")
            cleanup_temp_responses()
            return result
        
        result['pdf_processed'] = True
        logger.info(f"Successfully extracted data from PDF: {len(financial_data['pages'])} pages, {len(financial_data['tables'])} tables")
        
        # Save the extracted data
        if output_path:
            # Use the provided path
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(financial_data, f, indent=2)
            result['data_path'] = output_path
        else:
            # Use the default storage location
            data_path = save_pdf_data(pdf_path, financial_data)
            result['data_path'] = data_path
        
        result['success'] = result['data_path'] is not None
        
        # Clean up temporary files
        cleanup_temp_responses()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in extracting and saving PDF data: {str(e)}")
        result['errors'].append(str(e))
        cleanup_temp_responses()
        return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process financial PDF to Google Sheets")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--spreadsheet", help="Google Sheets spreadsheet ID")
    parser.add_argument("--query", help="User query to focus the update")
    parser.add_argument("--sheets", help="Comma-separated list of sheet names to update")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--extract-only", action="store_true", help="Extract and save data without updating sheets")
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction even if data exists")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Process sheet names if provided
    sheet_names = None
    if args.sheets:
        sheet_names = [name.strip() for name in args.sheets.split(',')]
    
    if args.extract_only:
        # Extract and save data only
        results = extract_and_save_pdf_data(args.pdf_path, args.output)
    else:
        # Process PDF and update sheets
        results = process_financial_pdf_to_sheets(
            args.pdf_path,
            args.spreadsheet,
            args.query,
            sheet_names,
            reuse_data=not args.force_extract
        )
    
    # Output results
    if args.output and not args.extract_only:  # If extract_only, output is used for data
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2)) 