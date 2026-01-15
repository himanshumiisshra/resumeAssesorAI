from flask import Flask, json, request, jsonify
from dotenv import load_dotenv
import os
import logging
from flask_cors import CORS
import google.generativeai as genai
import io
import fitz  # PyMuPDF for text extraction

# Load env variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

AUTH_SECRET = os.getenv("AUTH_SECRET")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

if not GOOGLE_GEMINI_API_KEY:
    raise EnvironmentError("GOOGLE_GEMINI_API_KEY is missing. Please check your .env file.")

# Configure Gemini
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# ✅ UPDATED: Using the generic alias. 
# This points to the stable Flash version (usually 1.5) which has a working free tier.
GEMINI_MODEL = "gemini-flash-latest"
model = genai.GenerativeModel(GEMINI_MODEL)

logging.info(f"Using Gemini model: {GEMINI_MODEL}")

# Flask app
app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "your BACKEND AI is LIVE"


def generate_prompt(job_description=None):
    base_prompt = """
        You are an advanced AI model designed to analyze the compatibility between a CV and a job description. 
        Your task is to output a structured JSON format.
    """

    if job_description:
        prompt = f"""
        {base_prompt}
        Here is the Job Description: {job_description}
        The CV is attached for analysis. Analyze the CV against the job description and provide detailed insights.
        
        Output valid JSON with these keys:
        - matching_analysis: Detailed analysis of strengths and gaps.
        - description: A brief summary.
        - score: Integer (0-100) based on qualifications.
        - skill_match_score: Integer (0-100) specifically for hard skills.
        - recommendation: Actionable suggestions for improvement.
        """
    else:
        prompt = f"""
        {base_prompt}
        The CV is attached for analysis. No job description was provided, so analyze the CV generally.
        
        Output valid JSON with these keys:
        - matching_analysis: Detailed analysis.
        - description: General summary.
        - score: Integer (0-100).
        - recommendation: Suggestions for improvement.
        """
    return prompt


def extract_text_from_pdf(pdf_bytes):
    """Extract plain text from PDF bytes using PyMuPDF."""
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"PDF extraction error: {e}")
        raise


@app.route("/upload-resume", methods=["POST"])
def uploadResume():
    try:
        # 🔐 Authorization
        auth_secret_fetched = (
            request.headers.get("Authorization")
            or request.headers.get("authorization")
            or request.json.get("authorization")
            or request.json.get("Authorization")
        )
        
        # Verify Auth (Uncomment strict check for production)
        if not auth_secret_fetched or auth_secret_fetched != AUTH_SECRET:
             # logging.warning("Auth failed or skipped for testing") 
             if auth_secret_fetched != AUTH_SECRET:
                 return jsonify({"error": "Invalid authorization secret."}), 401

        # 📄 Resume file
        job_description = request.form.get("job_description")
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400

        # ✅ Extract text
        pdf_bytes = file.read()
        extracted_text = extract_text_from_pdf(pdf_bytes)
        if not extracted_text.strip():
            return jsonify({"error": "Could not extract text from PDF."}), 400

        logging.info(f"Extracted PDF Text (first 500 chars): {extracted_text[:500]}")

        # 🤖 Generate prompt
        prompt = generate_prompt(job_description=job_description)
        
        logging.info(f"Sending request to model: {GEMINI_MODEL}")
        
        # ✅ UPDATED: Use Native JSON Mode
        response = model.generate_content(
            f"{prompt}\nHere is the resume text:\n{extracted_text}",
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )

        summary_text = response.text.strip()
        
        # Parse the JSON
        try:
            summary_json = json.loads(summary_text)
        except Exception as parse_error:
            logging.error(f"Error parsing JSON: {parse_error}")
            return jsonify(
                {
                    "error": "Model failed to return valid JSON.",
                    "raw_response": summary_text
                }
            ), 500

        return jsonify(
            {
                "extracted_text_preview": extracted_text[:1000], 
                "summary": summary_json,
            }
        )

    except Exception as e:
        logging.error(f"Error in uploadResume: {e}")
        return jsonify({"error": f"Unexpected error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)