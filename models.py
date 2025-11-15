import re
import cv2
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class OCR:
    def preprocess_image(self, image_path, scale_percent=200):
        self.scale_percent = scale_percent
        img = cv2.imread(image_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        width = int(gray_img.shape[1] * self.scale_percent / 100)
        height = int(gray_img.shape[0] * self.scale_percent / 100)
        gray_img = cv2.resize(gray_img, (width, height), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.fastNlMeansDenoising(gray_img, h=10)
        return denoised

    def clean_text(self, text):
        cleaned = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9\s.,;:!?()%\-—–«»]', '', text)
        return cleaned

    def img2txt(self, img, min_confidence=60):
        self.min_confidence = min_confidence
        processed = self.preprocess_image(img)
        config = '--psm 6 --oem 3'
        data = pytesseract.image_to_data(
            processed,
            lang='rus+eng',
            config=config,
            output_type=pytesseract.Output.DICT
        )
        filtered_text = []
        for i, conf in enumerate(data['conf']):
            if int(conf) > self.min_confidence:
                filtered_text.append(data['text'][i])
        return ' '.join(filtered_text)

    def process(self, image_path):
        text = self.img2txt(image_path)
        return self.clean_text(text)


class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def describe(self, image_path, max_length=50):
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=max_length)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return caption

    def describe_with_context(self, image_path, question):
        image = Image.open(image_path)
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs)
        answer = self.processor.decode(output[0], skip_special_tokens=True)

        return answer