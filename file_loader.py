import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from models import OCR, ImageCaptioner
import textract
from docx import Document
from docx2pdf import convert


class PDFProcessor:
    def __init__(self, ocr_model=None, captioner_model=None, min_text_length=20):
        self.ocr = ocr_model or OCR()
        self.captioner = captioner_model or ImageCaptioner()
        self.temp_dir = tempfile.mkdtemp()
        self.min_text_length = min_text_length

    def convert_to_pdf(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return file_path
        elif ext == ".docx":
            pdf_path = file_path.replace(".docx", ".pdf")
            convert(file_path, pdf_path)
            return pdf_path
        elif ext == ".doc":
            # DOC2DOCX
            docx_path = file_path.replace(".doc", ".docx")
            text = textract.process(file_path, encoding="utf-8").decode("utf-8")
            doc = Document()
            doc.add_paragraph(text)
            doc.save(docx_path)
            # DOCX2PDF
            pdf_path = docx_path.replace(".docx", ".pdf")
            convert(docx_path, pdf_path)
            return pdf_path
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def process_pdf(self, file_path):
        pdf_path = self.convert_to_pdf(file_path)
        loader = PyPDFLoader(pdf_path, extract_images=True)
        pages = loader.load()
        processed_pages = []
        for page in pages:
            if hasattr(page, "images") and page.images:
                for img_idx, img_data in enumerate(page.images):
                    img_path = os.path.join(self.temp_dir, f"page_{page.metadata['page']}_img_{img_idx}.png")
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    ocr_text = self.ocr.process(img_path)
                    if len(ocr_text.strip()) >= self.min_text_length:
                        replacement = ocr_text
                    else:
                        caption = self.captioner.describe(img_path)
                        replacement = caption
                    page.page_content = page.page_content.replace(f"[IMG:{img_idx}]", replacement)
            processed_pages.append(page)
        return processed_pages

    def cleanup(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
