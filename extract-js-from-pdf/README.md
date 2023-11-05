# extract-js-from-pdf

## Pre-requisites

1. Node.js 18+
2. Python 3.8+
3. Poetry (Optional)

## Installation

1. Clone the repository
2. Run `npm install` to install the dependencies
3. Run `poetry install --no-root` to install the Python dependencies
   - If you don't have poetry installed, you can install the dependencies manually by running `pip install git+https://github.com/enzok/peepdf.git`

## Usage

1. Extract JavaScript from a PDF file by running `./extract-js-from-pdf.sh <INPUT_DIR> <OUTPUT_DIR>`
   - `<INPUT_DIR>` is the directory containing the PDF files
   - `<OUTPUT_DIR>` is the directory where the extracted JavaScript files will be saved
