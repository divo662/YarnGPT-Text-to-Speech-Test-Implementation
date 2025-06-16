# YarnGPT Text-to-Speech Test Implementation

A test implementation of the YarnGPT model for Text-to-Speech (TTS) conversion. This project serves as a proof of concept and testing ground for the YarnGPT model's capabilities in speech synthesis.

## 🌟 Features

- **YarnGPT Model Integration**: Test implementation of the YarnGPT model for TTS
- **Modern Web Interface**: Clean, responsive design with real-time feedback
- **Basic Audio Processing**: Simple audio generation and playback
- **Performance Monitoring**: Basic tracking of generation times and performance
- **Rate Limited API**: Protected endpoints with usage limits
- **Audio Download**: Save generated audio files locally

## 🛠️ Technical Stack

### Backend
- Python 3.x
- Flask (Web Framework)
- PyTorch (Deep Learning)
- YarnGPT Model
- FFmpeg (Audio Processing)

### Frontend
- HTML5
- CSS3 (Modern styling with CSS variables)
- JavaScript (Vanilla)
- Font Awesome (Icons)

## 📋 Prerequisites

- Python 3.x
- FFmpeg installed on your system
- Git
- Virtual environment (recommended)
- YarnGPT model files

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/voice-yarngpt.git
cd voice-yarngpt
```

2. Set up the backend:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
   - Create a `.env` file in the backend directory
   - Add necessary configuration (see Configuration section)

4. Start the backend server:
```bash
python app.py
```

5. Serve the frontend:
   - Open `frontend/index.html` in your web browser
   - Or use a local server of your choice

## ⚙️ Configuration

Create a `.env` file in the backend directory with the following variables:

```env
FLASK_ENV=development
FLASK_APP=app.py
SECRET_KEY=your_secret_key
```

## 🎯 Usage

1. Open the web interface in your browser
2. Enter or paste the text you want to convert to speech
3. Click "Generate" to start the conversion
4. Wait for the processing to complete
5. Play the generated audio or download it

## 🔒 API Endpoints

- `POST /generate`: Generate single audio file
- `POST /batch-generate`: Generate multiple audio files
- `GET /health`: Health check endpoint

## 🛡️ Rate Limiting

- Single generation: 20 requests per hour
- Batch generation: 10 requests per hour
- Health check: 10 requests per minute

## 📁 Project Structure

```
voice-yarngpt/
├── backend/
│   ├── app.py              # Main application file
│   ├── config.py           # Configuration settings
│   ├── requirements.txt    # Python dependencies
│   └── yarngpt/           # YarnGPT model implementation
├── frontend/
│   └── index.html         # Web interface
└── README.md
```

## ⚠️ Important Note

This is a test implementation of the YarnGPT model for Text-to-Speech conversion. The project is intended for:
- Testing the YarnGPT model's capabilities
- Evaluating performance and quality
- Understanding the model's behavior
- Gathering feedback for potential improvements

## ⚠️ Known Issues

- Model performance may vary based on input text
- Processing time may be longer for complex inputs
- Audio quality may need optimization

## 🔮 Future Improvements

- [ ] Optimize model performance
- [ ] Improve audio quality
- [ ] Add more voice options
- [ ] Implement real-time voice preview
- [ ] Add voice customization options

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## 🙏 Acknowledgments

- YarnGPT model developers
- Open-source community
- All contributors to this test implementation

---

Made with ❤️ for testing YarnGPT voice capabilities 