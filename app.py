import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from dotenv import load_dotenv
import re
from markupsafe import Markup
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import markdown

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")

# Initialize the summarization model (only once at startup)
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add custom filter for newlines
@app.template_filter('nl2br')
def nl2br(value):
    if value:
        value = str(value).replace('\n', Markup('<br>'))
    return value

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def get_ai_summary(text):
    try:
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
            
        return " ".join(summaries)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Extract text based on file type
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
        
        return redirect(url_for('results'))
    
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

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
