#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the required modules
try:
    # Try to import from truffles package (for module usage)
    from truffles.pdf_processor import extract_financial_data_from_pdf
    from truffles.simple_mapper import map_financial_data, save_context_to_file
except ImportError:
    # Try to import directly (for direct script usage)
    try:
        from pdf_processor import extract_financial_data_from_pdf
        from simple_mapper import map_financial_data, save_context_to_file
    except ImportError:
        print("Error: Required modules not found. Make sure you're in the correct directory or install dependencies.")
        sys.exit(1)

def main():
    """
    Command-line interface for processing PDFs and mapping financial data to a matrix
    """
    parser = argparse.ArgumentParser(description='Process PDFs and map financial data')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract data from a PDF')
    extract_parser.add_argument('--pdf', required=True, help='Path to the PDF file')
    extract_parser.add_argument('--output', required=True, help='Path to save the extracted data')
    
    # Map command
    map_parser = subparsers.add_parser('map', help='Map financial data to a matrix')
    map_parser.add_argument('--data', required=True, help='Path to the extracted data JSON file')
    map_parser.add_argument('--matrix', required=True, help='Path to the matrix template JSON file')
    map_parser.add_argument('--output', required=True, help='Path to save the mapping results')
    map_parser.add_argument('--save-context', help='Path to save the formatted context (optional)')
    
    # Full flow command
    full_parser = subparsers.add_parser('full', help='Run the complete extraction and mapping flow')
    full_parser.add_argument('--pdf', required=True, help='Path to the PDF file')
    full_parser.add_argument('--matrix', required=True, help='Path to the matrix template JSON file')
    full_parser.add_argument('--output', required=True, help='Path to save the mapping results')
    full_parser.add_argument('--save-context', help='Path to save the formatted context (optional)')
    full_parser.add_argument('--save-extracted', help='Path to save the extracted data (optional)')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        # Extract data from PDF
        print(f"Extracting data from {args.pdf}...")
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
            return 1
        
        try:
            pdf_data = extract_financial_data_from_pdf(args.pdf, api_key=api_key)
            
            # Save the extracted data
            with open(args.output, 'w') as f:
                json.dump(pdf_data, f, indent=2)
            
            print(f"Data extracted and saved to {args.output}")
            print(f"Found {len(pdf_data.get('key_value_pairs', []))} key-value pairs and {len(pdf_data.get('tables', []))} tables")
            return 0
        except Exception as e:
            print(f"Error extracting data: {str(e)}")
            return 1
    
    elif args.command == 'map':
        # Map financial data to matrix
        print(f"Mapping data from {args.data} to matrix template {args.matrix}...")
        
        try:
            # Load financial data
            with open(args.data, 'r') as f:
                financial_data = json.load(f)
            
            # Load matrix template
            with open(args.matrix, 'r') as f:
                matrix = json.load(f)
            
            # Save context if requested
            if args.save_context:
                print(f"Saving context to {args.save_context}...")
                save_context_to_file(financial_data, args.save_context)
            
            # Map financial data to matrix
            print("Mapping data to matrix...")
            result = map_financial_data(financial_data, matrix)
            
            # Check for errors
            if 'error' in result:
                print(f"Error mapping data: {result['error']}")
                return 1
            
            # Save result
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Print stats
            stats = result.get('stats', {})
            print(f"Mapping completed successfully!")
            print(f"Results saved to {args.output}")
            print(f"Stats: {stats.get('cells_filled', 0)} cells filled, {stats.get('cells_not_filled', 0)} cells not filled")
            return 0
        except Exception as e:
            print(f"Error mapping data: {str(e)}")
            return 1
    
    elif args.command == 'full':
        # Run the complete flow
        print(f"Running complete flow: extract from {args.pdf} and map to matrix {args.matrix}...")
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("Error: OpenAI API key is required. Set the OPENAI_API_KEY environment variable.")
            return 1
        
        try:
            # Extract data from PDF
            print(f"Step 1: Extracting data from {args.pdf}...")
            pdf_data = extract_financial_data_from_pdf(args.pdf, api_key=api_key)
            print(f"Found {len(pdf_data.get('key_value_pairs', []))} key-value pairs and {len(pdf_data.get('tables', []))} tables")
            
            # Save extracted data if requested
            if args.save_extracted:
                with open(args.save_extracted, 'w') as f:
                    json.dump(pdf_data, f, indent=2)
                print(f"Extracted data saved to {args.save_extracted}")
            
            # Load matrix template
            print(f"Step 2: Loading matrix template from {args.matrix}...")
            with open(args.matrix, 'r') as f:
                matrix = json.load(f)
            
            # Save context if requested
            if args.save_context:
                print(f"Saving context to {args.save_context}...")
                save_context_to_file(pdf_data, args.save_context)
            
            # Map financial data to matrix
            print("Step 3: Mapping data to matrix...")
            result = map_financial_data(pdf_data, matrix)
            
            # Check for errors
            if 'error' in result:
                print(f"Error mapping data: {result['error']}")
                return 1
            
            # Save result
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Print stats
            stats = result.get('stats', {})
            print(f"Full process completed successfully!")
            print(f"Results saved to {args.output}")
            print(f"Stats: {stats.get('cells_filled', 0)} cells filled, {stats.get('cells_not_filled', 0)} cells not filled")
            return 0
        except Exception as e:
            print(f"Error in process: {str(e)}")
            return 1
    
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main()) 