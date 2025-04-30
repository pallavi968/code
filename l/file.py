# import os
# from llama_index.core import SimpleDirectoryReader
# from llama_index.readers.file import (
#     PDFReader,
#     DocxReader,
#     FlatReader,
 
   
 
# )
 
# # File extension to reader mapping
# file_extractors = {
#     ".pdf": PDFReader(),
#     ".docx": DocxReader(),
#     ".txt": FlatReader(),
   
# }
 
# def load_documents_from_directory(directory_path: str):
#     """
#     Loads documents from the given directory using the appropriate file extractors.
#     Only PDF, DOCX, and TXT files are supported at this stage.
#     """
#     return SimpleDirectoryReader(directory_path, file_extractor=file_extractors).load_data()

import os
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader, DocxReader, FlatReader

file_extractors = {
    ".pdf": PDFReader(),
    ".docx": DocxReader(),
    ".txt": FlatReader(),
}

def load_documents_from_directory(directory_path: str):
    """
    Loads documents from the given directory using the appropriate file extractors.
    Only PDF, DOCX, and TXT files are supported at this stage.
    """
    try:
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist.")
        return SimpleDirectoryReader(directory_path, file_extractor=file_extractors).load_data()
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return []