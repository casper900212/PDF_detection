import os
import pprint
from pdfid import pdfid

pp = pprint.PrettyPrinter(indent=4)

def analyze_pdfs_by_filenames(filenames):
    # 1. Setup
    options = pdfid.get_fake_options()
    options.scan = True
    options.json = True

    # 2. Actual analysis
    list_of_dict = pdfid.PDFiDMain(filenames, options)

    return list_of_dict

def analyze_pdfs_by_buffer(filenames, file_buffers):
    # 1. Setup
    options = pdfid.get_fake_options()
    options.scan = True
    options.json = True

    # 2. Actual analysis
    list_of_dict = pdfid.PDFiDMain(filenames, options, file_buffers)

    return list_of_dict

def disarm_pdfs_by_buffer(filenames, file_buffers):
    # 1. Setup
    options = pdfid.get_fake_options()

    options.disarm = False
    # If you want to return the disarmed buffer 
    # instead of the results dict, set this to True
    options.return_disarmed_buffer = True

    # 2. Actual analysis + disarm
    disarmed_pdf_buffers = pdfid.PDFiDMain(filenames, options, file_buffers)
    return disarmed_pdf_buffers

def main():
    # 1. Get PDF files in the directory
    directory_path = input("Input pdf directory: ")
    pdf_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]

    # 2. Read PDF files
    file_buffers = []
    print("reading files\n")
    print(len(pdf_files),"\n")
    for filename in pdf_files:
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                file_buffers.append(f.read())

    '''
    # 3. Analyze PDF from filenames
    print("STARTING ANALYSIS 1")
    results_1 = analyze_pdfs_by_filenames(pdf_files)
    pp.pprint(results_1)
    '''

    # 4. Analyze PDF from buffer
    # print("STARTING ANALYSIS 2")
    print("analyze pdf from buffer")
    results_2 = analyze_pdfs_by_buffer(pdf_files, file_buffers)
    # pp.pprint(results_2)

    '''
    # 5. Disarm PDF from buffer and return disarmed buffer
    print("STARTING DISARM")
    disarmed_pdf_buffers = disarm_pdfs_by_buffer(pdf_files, file_buffers)

    # 6. Analyze PDF from disarmed buffer
    print("STARTING DISARMED ANALYSIS")
    results_3 = analyze_pdfs_by_buffer(pdf_files, disarmed_pdf_buffers['buffers'])
    pp.pprint(results_3)
    '''

    import pandas as pd

    data = pd.DataFrame.from_dict(results_2)


    import csv

    fieldnames = ['version', 'filename', 'header', 'obj', 'endobj', 'stream', 'endstream', 'xref', 'trailer', 'startxref', '/Page', '/Encrypt', '/ObjStm', '/JS', '/JavaScript', '/AA', '/OpenAction', '/AcroForm', '/JBIG2Decode', '/RichMedia', '/Launch', '/EmbeddedFile', '/XFA', '/Colors > 2^24',
     '/JavaScript_hexcode_count', '/RichMedia_hexcode_count', '/EmbeddedFile_hexcode_count', '/ObjStm_hexcode_count', '/Encrypt_hexcode_count', '/OpenAction_hexcode_count', '/Page_hexcode_count','/JS_hexcode_count', '/AcroForm_hexcode_count', '/XFA_hexcode_count', '/Launch_hexcode_count', '/JBIG2Decode_hexcode_count', '/AA_hexcode_count']
    
    outputfile = input("Outputfile: ")
    with open(outputfile, 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        for row in data['reports']:
            writer.writerow(row)

        

if __name__ == '__main__':
    main()
    