from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import fitz
from PIL import Image
import pytesseract as pt
from llmlogic import generate
from concurrent.futures import ThreadPoolExecutor
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer

app = Flask(__name__)

# Set the path where uploads will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

translator = Translator()

# Initialize the tesseract reader
pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"


def translate_text(text):
    target_language = "en"
    try:
        translation = translator.translate(text, dest=target_language)
        if translation is not None and translation.text is not None:
            translated_text = translation.text
            return translated_text
        else:
            print("Translation failed. The translation text is None.")
            # Handle the failure accordingly
            return None  # Or raise an exception, return an error code, etc.
    except Exception as e:
        print(f"Translation error: {e}")
        # Handle the exception accordingly
        return None  # Or raise an exception, return an error code, etc.

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pt.image_to_string(img, lang="mar")
    return text

def process_image(image_path):
    # Extract text from the image using OCR
    text = extract_text_from_image(image_path)
    return text

def extract_text_from_pdf_page(page, output_folder):
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(os.path.join(output_folder, f"image_{page.number}.png"))

    # Extract text from the image using OCR
    text = extract_text_from_image(os.path.join(output_folder, f"image_{page.number}.png"))
    return text

def process_pdf(pdf_path, output_folder):
    doc = fitz.open(pdf_path)

    with ThreadPoolExecutor() as executor:
        # Extract images and text from each page concurrently
        futures = []
        for page in doc:
            futures.append(executor.submit(extract_text_from_pdf_page, page, output_folder))

        # Wait for all tasks to complete
        text_data = [future.result() for future in futures]

    doc.close()
    return text_data

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser also
        # submits an empty part without a filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Define the output folder for images
            output_folder = 'images'
            os.makedirs(output_folder, exist_ok=True)

            # Check if the uploaded file is a PDF or an image
            if file_path.lower().endswith(('.pdf')):
                # Extract text from the PDF using parallel processing
                extracted_text = process_pdf(file_path, output_folder)
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract text from the image
                extracted_text = process_image(file_path)
                print(extracted_text)

            # Extract text from the PDF using parallel processing
            # extracted_text = process_pdf(file_path, output_folder)

            text=extracted_text
            # engtext = translate_text(text)
            input_text1 = text
            # Initialize the parser and tokenizer
            input_text = ', '.join(input_text1)
            print(len(input_text))
            print(type(input_text))
            parser = PlaintextParser.from_string(input_text, Tokenizer("english"))

            # Initialize the LSA summarizer
            lsa_summarizer = LsaSummarizer()

            # Summarize the text
            summary = lsa_summarizer(parser.document, sentences_count=20)  # Adjust sentences_count as needed

            text = ""
            # Print the summary
            for sentence in summary:
                text += str(sentence) + " "
            print(len(text))

            generated_output = generate(text)
            sections = [section.strip() for section in generated_output.split('\n') if section.strip()]

            sections[-1] = sections[-1][:-4]

            return render_template('result.html', text_data=sections)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
