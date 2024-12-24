from zipfile import ZipFile

class ZipExtractor:

    def __init__(self, zip_path, extract_path):
        self.zip_path = zip_path
        self.extract_path = extract_path

    def extract(self):
        with ZipFile(self.zip_path, 'r') as zip_extract:
            zip_extract.extractall(self.extract_path)