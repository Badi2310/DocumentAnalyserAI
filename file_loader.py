import os
import tempfile
from pdf2image import convert_from_path
from pypdf import PdfReader
from docx import Document
from fpdf import FPDF


class PDFProcessor:
    def __init__(self, ocr_model=None, captioner_model=None, min_text_length=20):
        from models import OCR, ImageCaptioner

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
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])

            pdf = FPDF()
            pdf.add_page()
            pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
            pdf.set_font("DejaVu", size=12)
            for line in text.splitlines():
                pdf.cell(0, 10, text=line, new_x="LMARGIN", new_y="NEXT")
            pdf.output(pdf_path)
            return pdf_path
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def has_meaningful_text(self, text):
        """Check letter variety"""
        clean = text.strip()
        letters = sum(c.isalpha() for c in clean)
        return (
            len(clean) >= self.min_text_length and letters > self.min_text_length * 0.5
        )

    def image_extracter(self, page, page_num):
        """Extract images"""
        images_data = []

        try:
            for img_idx, image_file_object in enumerate(page.images):
                img_path = os.path.join(
                    self.temp_dir,
                    f"page_{page_num}_img_{img_idx}.{image_file_object.name.split('.')[-1]}",
                )

                with open(img_path, "wb") as f:
                    f.write(image_file_object.data)

                caption = self.captioner.describe(img_path)

                if caption:
                    images_data.append(
                        {"index": img_idx, "caption": caption, "path": img_path}
                    )

        except Exception as e:
            print(f"Error extracting images from page {page_num}: {e}")

        return images_data

    def format_images_text(self, images_data):
        """
        Convert image dictionary 2 string
        """
        if not images_data:
            return ""

        return "".join(
            [f"\n[IMAGE {img['index']}: {img['caption']}]" for img in images_data]
        )

    def process(self, file_path):
        """ "
        Full processing of file
        """
        pdf_path = self.convert_to_pdf(file_path)
        processed_pages = []

        try:
            reader = PdfReader(pdf_path)
            try:
                pages_images = convert_from_path(pdf_path, dpi=200)
            except Exception as e:
                print(f"Warning: Could not convert to images: {e}")
                pages_images = []

            for page_num, page in enumerate(reader.pages):
                print(f"\nProcessing page {page_num + 1}.")
                text = page.extract_text()

                images_data = self.image_extracter(page, page_num)
                images_text = self.format_images_text(images_data)

                if self.has_meaningful_text(text):
                    final_text = text + images_text
                    processed_pages.append(final_text)
                    print(
                        f"  Used extracted text ({len(text)} chars) + {len(images_data)} images"
                    )

                else:
                    if page_num < len(pages_images):
                        img_path = os.path.join(
                            self.temp_dir, f"page_{page_num}_ocr.png"
                        )
                        pages_images[page_num].save(img_path, "PNG")

                        ocr_text = self.ocr.process(img_path)

                        if self.has_meaningful_text(ocr_text):
                            final_text = ocr_text + images_text
                            processed_pages.append(final_text)
                            print(
                                f"  Used OCR ({len(ocr_text)} chars) + {len(images_data)} images"
                            )
                        else:
                            caption = self.captioner.describe(img_path)
                            if caption:
                                final_text = (
                                    f"[FULL PAGE IMAGE: {caption}]" + images_text
                                )
                            else:
                                final_text = "[FULL PAGE IMAGE]" + images_text
                            processed_pages.append(final_text)
                            print(
                                f"  Used full page captioning + {len(images_data)} images"
                            )
                    else:
                        processed_pages.append(f"[NO TEXT EXTRACTED]{images_text}")
                        print(f"  No text, only {len(images_data)} embedded images")

        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

        return processed_pages

    def cleanup(self):
        try:
            for file in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
            print("Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {e}")


# Example of usage

# path = "1.pdf"
# loader = PDFProcessor()
# try:
#     data = loader.process(file_path=path)
#     text = "\n".join(data)
#     with open("result.txt", "w", encoding="utf-8") as f:
#         f.write(text)
#     print(f"\nProcessed {len(data)} pages")
# finally:
#     loader.cleanup()
