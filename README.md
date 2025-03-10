# Project Rosewood

A web-based application that utilizes artificial intelligence to generate summaries of PDF and Word documents, as well as create interactive quizzes based on their content. This application operates completely offline, utilizing Hugging Face Transformers models.



## Features

- Upload PDF and Word documents
- Extract text from documents
- Generate AI-powered summaries using Hugging Face Transformers
- Create interactive quizzes based on document content
- Clean, responsive user interface
- No API keys or internet connection required after initial setup
- Customizable quiz settings (Coming soon!!)
- Option to view answers 
- Option to download summary

## Setup Instructions (important!)

### Prerequisites

- Python 3.7 or higher
- Sufficient disk space for model files (approximately 1.5GB)

### Installation

1. Clone or download this repository
2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. The first time you run the application, it will automatically download the necessary model files from Hugging Face. After that, the application will work offline.

### Running the Application

1. Start the Flask server:

```
python app.py
```

2. Open your web browser and navigate to:

```
http://127.0.0.1:5000
```

## Usage

1. On the homepage, click the "Choose File" button to select a PDF or Word document
2. Click "Upload & Process" to submit the document
3. Wait for the AI to process your document (this may take a few moments)
4. View the summary tab to see a concise overview of your document
5. Switch to the quiz tab to test your knowledge with AI-generated questions
6. Use the "Show/Hide Answers" button to toggle answer visibility

## Technologies Used

- Flask: Web framework
- PyPDF2: PDF text extraction
- python-docx: Word document text extraction
- Hugging Face Transformers: AI-powered summarization and text processing
- PyTorch: Deep learning framework for running the AI models
- Bootstrap: Frontend styling
- Markdown: For formatting the summary

## How It Works

### Summarization
The application uses the `facebook/bart-large-cnn` model from Hugging Face, which is specifically trained for text summarization. For longer documents, the text is split into chunks and each chunk is summarized separately, then combined.
The application uses the `facebook/bart-large-cnn` model from Hugging Face, which is specifically trained for text summarization. For longer documents, the text is split into chunks and each chunk is summarized separately, then combined.

### Quiz Generation
The application uses a custom algorithm to:
1. Extract important sentences from the document
2. Identify keywords to create fill-in-the-blank style questions
3. Generate plausible distractors (wrong answers) from other parts of the text
4. Format everything into a clean multiple-choice quiz

## Advantages Over API-Based Solutions

- No API costs or rate limits
- Works offline without internet connection after initial model download
- More privacy since your documents aren't sent to external servers
- Fully customizable - you can swap in different models if needed

## Limitations

- Initial download of models may take some time depending on your internet connection
- Processing very large documents may be slower compared to cloud-based solutions
- Requires more system resources (RAM and disk space) than API-based alternatives

## Future Enhancements

- Support for more document formats
- Custom quiz generation options
- Document comparison functionality
- Save and export summaries and quizzes
- Option to use smaller, faster models for resource-constrained systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing the `facebook/bart-large-cnn` model

