from flask import Flask, json, request, jsonify, Response
from groq import Groq
from dotenv import load_dotenv
import os
import logging
from flask_cors import CORS
import google.generativeai as genai
import io
import fitz  # PyMuPDF for text extraction

# Load env variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

AUTH_SECRET = os.getenv('AUTH_SECRET')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY is missing. Please check your .env file.")

GOOGLE_GEMINI_API_KEY = os.getenv('GOOGLE_GEMINI_API_KEY')
if not GOOGLE_GEMINI_API_KEY:
    raise EnvironmentError("GOOGLE_GEMINI_API_KEY is missing. Please check your .env file.")

# Configure Gemini
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Pick model from env or default
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro-latest")
model = genai.GenerativeModel(GEMINI_MODEL)

logging.info(f"Using Gemini model: {GEMINI_MODEL}")

# Configure Groq
client = Groq(api_key=GROQ_API_KEY)

# Flask app
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return 'your BACKEND AI is LIVE'

def generate_prompt(job_description=None):
    base_prompt = """
        You are an advanced AI model designed to analyze the compatibility between a CV and a job description. 
        Your task is to output a structured JSON format that includes the following:
        
        1. matching_analysis: Analyze the CV against the job description to identify key strengths and gaps.
        2. description: Summarize the relevance of the CV to the job description in a few concise sentences.
        3. score: Provide a numerical compatibility score (0-100) based on qualifications, skills, and experience.
        4. recommendation: Suggest actions for the candidate to improve their match or readiness for the role.
    """

    if job_description:
        prompt = f"""
        {base_prompt}
        Here is the Job Description: {job_description}
        The CV is attached for analysis. Analyze the CV against the job description and provide detailed insights.
        Your output must be in JSON format as follows:
        {{
          "matching_analysis": "Your detailed analysis here.",
          "description": "A brief summary here.",
          "score": 85,
          "skill_match_score": "The skill match score with the required skills in job description. Out of 100. Only number here",
          "recommendation": "Your suggestions here."
        }}
        """
    else:
        prompt = f"""
        {base_prompt}
         The CV is attached for analysis. Analyze the CV and provide detailed insights. 
        As no job description is provided, analyze the CV in general and suggest areas for improvement.
        Your output must be in JSON format as follows:
        {{
          "matching_analysis": "Your detailed analysis here.",
          "description": "A general summary here.",
          "score": 70,
          "recommendation": "Your suggestions here."
        }}
        """
    return prompt


def extract_text_from_pdf(pdf_bytes):
    """Extract plain text from PDF bytes using PyMuPDF."""
    pdf_stream = io.BytesIO(pdf_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()


@app.route('/upload-resume', methods=['POST'])
def uploadResume():
    auth_secret_fetched = request.headers.get('Authorization') or request.headers.get('authorization') \
                          or request.json.get('authorization') or request.json.get('Authorization')
    if not auth_secret_fetched:
        return jsonify({'error': 'Authorization header is required.'}), 401
    
    if auth_secret_fetched != AUTH_SECRET:
        return jsonify({'error': 'Invalid authorization secret.'}), 401
    
    job_description = request.form.get('job_description')

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        try:
            pdf_bytes = file.read()

            # ✅ Extract text before sending to Gemini
            extracted_text = extract_text_from_pdf(pdf_bytes)
            logging.info(f"Extracted PDF Text (first 500 chars): {extracted_text[:500]}")

        except Exception as e:
            logging.error(f"Error reading or extracting PDF: {e}")
            return jsonify({"error": f"Error reading or extracting PDF: {e}"}), 500

        try:
            prompt = generate_prompt(job_description=job_description)

            # Send extracted text instead of PDF file object
            response = model.generate_content(f"{prompt}\nHere is the resume text:\n{extracted_text}")
            summary = response.text.strip()
            
            if summary.startswith("```json") and summary.endswith("```"):
                summary = summary[7:-3].strip()
            
            try:
                summary_json = json.loads(summary)
            except Exception as parse_error:
                logging.error(f"Error parsing JSON: {parse_error}")
                return jsonify({"error": f"Error parsing JSON from model output: {parse_error}", 
                                "raw_response": summary}), 500

            # ✅ Return extracted text as well for debugging
            return jsonify({
                "extracted_text_preview": extracted_text[:1000],  # first 1000 chars
                "summary": summary_json
            })
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return jsonify({"error": f"Error generating summary: {e}"}), 500


# genie route remains unchanged ...

if __name__ == '__main__':
    app.run(debug=True)
