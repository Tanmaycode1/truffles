An AI-powered tool that extracts financial data from PDF documents and updates Google Sheets automatically.

## Features

- **PDF Extraction**: Extract structured and semi-structured data from financial PDF documents
- **Google Sheets Integration**: Automatically update Google Sheets with extracted data
- **Intelligent Mapping**: Match financial items from PDFs to the appropriate cells in Google Sheets
- **Natural Language Queries**: Ask questions about the extracted data or request specific updates
- **Multi-language Support**: Handle financial terms in different languages
- **User-friendly Interface**: Simple web UI for uploading PDFs, connecting to sheets, and querying data

## Getting Started

<img width="1470" alt="Screenshot 2025-04-11 at 4 06 28 AM" src="https://github.com/user-attachments/assets/0a608622-9416-4c18-91a6-6c7560525ce4" />

<img width="1470" alt="Screenshot 2025-04-11 at 4 06 15 AM" src="https://github.com/user-attachments/assets/874bd78a-bafa-45b8-8fa2-2d6e4e59db5c" />

<img width="1470" alt="Screenshot 2025-04-11 at 4 06 05 AM" src="https://github.com/user-attachments/assets/9d577097-fbb3-45c8-8a63-bbef00b921d7" />

<img width="1470" alt="Screenshot 2025-04-11 at 4 03 27 AM" src="https://github.com/user-attachments/assets/5240c8db-89fc-4492-80b5-cba004972143" />

### Prerequisites

- Python 3.8 or higher
- Google Cloud Platform account with Google Sheets API enabled
- OpenAI API key (optional, for enhanced mapping and query handling)
- Poppler (for pdf2image library):
  - On macOS: `brew install poppler`
  - On Ubuntu/Debian: `apt-get install poppler-utils`
  - On Windows: Install from http://blog.alivate.com.au/poppler-windows/

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/financial-pdf-to-sheets.git
   cd financial-pdf-to-sheets
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Google credentials:
   - Create a project in Google Cloud Console
   - Enable the Google Sheets API
   - Create OAuth credentials or a service account
   - Download credentials file as `credentials.json` or `service_account.json` and place in project root

4. Set environment variables:
   ```
   # Create a .env file with:
   OPENAI_API_KEY=your_openai_api_key
   SECRET_KEY=your_flask_secret_key
   ```

### Running the Application

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open a web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Follow the steps in the UI:
   - Upload a financial PDF document
   - Provide a Google Sheet URL
   - View the extracted data and mapped fields
   - Use natural language queries to interact with the data

## How It Works

1. **PDF Processing**: Uses multiple libraries (PyPDF2, pdfplumber, tabula) to extract both structured tables and key-value pairs from financial PDFs.

2. **Sheet Structure Analysis**: Analyzes the Google Sheet to understand its structure, including identifying if it's a transposed sheet (common in financial statements).

3. **Intelligent Mapping**: Uses string similarity and pattern matching to map extracted financial items to the appropriate cells in the sheet.

4. **Natural Language Understanding**: Processes user queries to either retrieve information or update specific cells.

5. **Google Sheets API**: Securely updates Google Sheets with the mapped financial data.

## Example Queries

- "What is the total revenue?"
- "Fill the net income in the income statement"
- "What were the total assets in 2021?"
- "Update the depreciation value in the balance sheet"

## Supported Financial Statements

- Balance Sheets
- Income Statements
- Cash Flow Statements
- Statements of Changes in Equity

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for natural language processing capabilities
- Google for Sheets API
- PyPDF2, pdfplumber, and tabula-py for PDF processing 
