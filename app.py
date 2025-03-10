import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, redirect, url_for, flash, session, current_app
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from dotenv import load_dotenv
import re
from markupsafe import Markup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import markdown
from functools import lru_cache
import time

# Load environment variables
load_dotenv()


# Initialize Flask application
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Project Rosewood startup')

# Initialize the summarization model (only once at startup)
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    app.logger.info('Summarization model loaded successfully')
except Exception as e:
    app.logger.error(f'Error loading summarization model: {str(e)}')
    summarizer = None

# Configure upload folder and file size limits
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Add custom filter for newlines
@app.template_filter('nl2br')
def nl2br(value):
    if value:
        value = str(value).replace('\n', Markup('<br>'))
    return value

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lru_cache(maxsize=32)
def extract_text_from_pdf(file_path):
    """Extract text from PDF with caching for better performance"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text
    except Exception as e:
        app.logger.error(f'Error extracting text from PDF {file_path}: {str(e)}')
        raise

@lru_cache(maxsize=32)
def extract_text_from_docx(file_path):
    """Extract text from DOCX with caching for better performance"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        app.logger.error(f'Error extracting text from DOCX {file_path}: {str(e)}')
        raise

def get_ai_summary(text):
    """Generate AI summary with error handling and logging"""
    if not summarizer:
        app.logger.error('Summarizer model not available')
        return "Error: Summarization model not available"

    try:
        start_time = time.time()
        
        # Split text into chunks if it's too long
        max_chunk_length = 1024
        chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
        
        summaries = []
        for chunk in chunks[:3]:  # Process up to 3 chunks to avoid memory issues
            if len(chunk.strip()) > 100:  # Only summarize substantial chunks
                summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                summaries.append(summary)
        
        if not summaries:
            return "Text too short or couldn't be summarized."
        
        final_summary = " ".join(summaries)
        processing_time = time.time() - start_time
        app.logger.info(f'Summary generated in {processing_time:.2f} seconds')
        
        return final_summary
    except Exception as e:
        app.logger.error(f'Error generating summary: {str(e)}')
        return f"Error generating summary: {str(e)}"

def cleanup_old_files():
    """Clean up old uploaded files to prevent disk space issues"""
    try:
        current_time = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Remove files older than 1 hour
            if os.path.getmtime(file_path) < current_time - 3600:
                os.remove(file_path)
                app.logger.info(f'Cleaned up old file: {filename}')
    except Exception as e:
        app.logger.error(f'Error cleaning up files: {str(e)}')

@app.before_request
def before_request():
    """Perform cleanup operations before each request"""
    cleanup_old_files()

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            app.logger.info(f'File uploaded successfully: {filename}')
            
            # Extract text based on file type
            start_time = time.time()
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif filename.endswith(('.docx', '.doc')):
                text = extract_text_from_docx(file_path)
            else:
                flash('Unsupported file format')
                return redirect(url_for('index'))
            
            # Generate summary and quiz
            summary = get_ai_summary(text)
            quiz = generate_quiz(text)
            
            # Store in session
            session['summary'] = summary
            session['quiz'] = quiz
            session['filename'] = filename
            
            processing_time = time.time() - start_time
            app.logger.info(f'File processed in {processing_time:.2f} seconds: {filename}')
            
            return redirect(url_for('results'))
            
        except Exception as e:
            app.logger.error(f'Error processing file {file.filename}: {str(e)}')
            flash('Error processing file. Please try again.')
            return redirect(url_for('index'))
    
    flash('File type not allowed')
    return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'summary' not in session or 'quiz' not in session:
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    return render_template('results.html', 
                         filename=session.get('filename', ''),
                         summary=session.get('summary', ''),
                         quiz=session.get('quiz', ''))

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return {
        'status': 'healthy',
        'summarizer_loaded': summarizer is not None,
        'upload_folder': os.path.exists(UPLOAD_FOLDER)
    }

def generate_quiz(text):
    try:
        # Load a model specifically for question generation
        # We'll use a simple approach: extract key sentences and create questions from them
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out short sentences and select a subset
        content_sentences = [s for s in sentences if len(s.split()) > 10]
        selected_sentences = content_sentences[:10]  # Take up to 10 substantial sentences
        
        # Create HTML directly
        quiz_questions = []
        
        # For each selected sentence, create a question
        question_count = min(5, len(selected_sentences))
        for i in range(question_count):
            if i >= len(selected_sentences):
                break
                
            sentence = selected_sentences[i]
            words = sentence.split()
            
            # Find a keyword to ask about
            keywords = [w for w in words if len(w) > 5 and w.isalpha()]
            if not keywords:
                continue
                
            keyword = keywords[0]
            question = sentence.replace(keyword, "________")
            
            # Create distractors (wrong answers)
            distractors = []
            for j in range(len(selected_sentences)):
                if j != i:
                    other_words = [w for w in selected_sentences[j].split() if len(w) > 4 and w.isalpha() and w != keyword]
                    if other_words:
                        distractors.append(other_words[0])
            
            while len(distractors) < 3 and len(keywords) > 1:
                for k in range(1, len(keywords)):
                    if keywords[k] not in distractors:
                        distractors.append(keywords[k])
                        break
            
            # Fill in with generic distractors if needed
            generic_distractors = ["option", "choice", "alternative", "selection"]
            while len(distractors) < 3:
                for gd in generic_distractors:
                    if gd not in distractors:
                        distractors.append(gd)
                        break
            
            # Limit to 3 distractors
            distractors = distractors[:3]
            
            # Add to questions list
            quiz_questions.append({
                'number': i+1,
                'question_text': question,
                'correct_answer': keyword,
                'distractors': distractors
            })
        
        if not quiz_questions:
            return Markup("<div class='quiz-container'><p>Unable to generate questions from the provided text.</p></div>")
        
        # Now render the quiz HTML
        return render_template('quiz_template.html', questions=quiz_questions)
    except Exception as e:
        return Markup(f"<div class='quiz-container'><p>Error generating quiz: {str(e)}</p></div>")

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true',
           use_reloader=False,
           host='0.0.0.0',
           port=int(os.getenv('PORT', 5000)))
