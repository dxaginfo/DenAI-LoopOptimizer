# LoopOptimizer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0-green.svg)

A specialized media automation tool designed to identify, analyze, and optimize loop points in audio and video content. It uses machine learning to detect optimal transition points for seamless looping, enabling creators to generate high-quality looping media for applications in gaming, digital signage, web backgrounds, and interactive media.

## Features

- **Automated Loop Detection**: Intelligently identify optimal loop points in media files
- **Crossfade Generation**: Create smooth transitions between loop start and end points
- **Quality Analysis**: Evaluate loop quality with detailed metrics and suggestions
- **Batch Processing**: Process multiple files with consistent settings
- **Cloud Integration**: Deploy as a cloud function with Google Cloud
- **Gemini API**: Leverage Google's Gemini API for enhanced pattern recognition

## Getting Started

### Prerequisites

- Python 3.9+
- FFmpeg 4.4+
- NumPy 1.20+
- SciPy 1.7+
- PyTorch 1.10+ or TensorFlow 2.7+
- Google Cloud SDK (for cloud deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/dxaginfo/DenAI-LoopOptimizer.git
cd DenAI-LoopOptimizer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Google Cloud credentials (if using cloud features)
gcloud auth application-default login
```

### Basic Usage

#### Command Line Interface

```bash
# Process a video file
python loop_optimizer.py --input video.mp4 --output looped_video.mp4 --media-type video --optimization high

# Process an audio file
python loop_optimizer.py --input audio.mp3 --output looped_audio.mp3 --media-type audio --crossfade 0.3
```

#### Python API

```python
from loop_optimizer import LoopOptimizer

# Initialize the optimizer
optimizer = LoopOptimizer()

# Process a media file
result = optimizer.process(
    media_source="path/to/video.mp4", 
    media_type="video",
    optimization_level="high"
)

# Access the results
print(f"Optimized media available at: {result.optimized_media}")
print(f"Loop points: {result.loop_points}")
print(f"Quality score: {result.loop_points[0].quality_score}")
```

## Architecture

### Components

1. **Core Engine**: Python-based analysis engine with ML capabilities
2. **Media Processor**: FFmpeg wrapper for handling various media formats
3. **API Layer**: REST API for programmatic access
4. **Frontend**: Optional static HTML interface
5. **Storage**: Google Cloud Storage integration for media files

### API Reference

LoopOptimizer provides a RESTful API that can be accessed at `/api/v1/optimize`:

```
POST /api/v1/optimize
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "mediaSource": "https://storage.googleapis.com/media-bucket/input.mp4",
  "mediaType": "video",
  "optimizationLevel": "high",
  "outputFormat": "mp4"
}
```

For full API documentation, see [API.md](docs/API.md).

## Deployment Options

### Local Deployment

Run the application locally for development or personal use:

```bash
# Start the API server
python server.py
```

### Google Cloud Deployment

Deploy as a Cloud Function:

```bash
gcloud functions deploy loop_optimizer \
  --runtime python39 \
  --trigger-http \
  --allow-unauthenticated \
  --memory 2048MB \
  --timeout 540s
```

Or deploy as a containerized service with Cloud Run:

```bash
# Build the container
gcloud builds submit --tag gcr.io/YOUR_PROJECT/loop-optimizer

# Deploy to Cloud Run
gcloud run deploy loop-optimizer \
  --image gcr.io/YOUR_PROJECT/loop-optimizer \
  --platform managed \
  --memory 2G \
  --timeout 10m
```

## Integration Examples

### Google Workspace Integration

```python
from loop_optimizer.integrations import GoogleDriveConnector

# Initialize the Google Drive connector
drive = GoogleDriveConnector(credentials_path="credentials.json")

# Upload a file for processing
file_id = drive.upload_file("local_video.mp4")

# Process the file
result = drive.process_file(file_id, optimization_level="medium")

# Download the result
drive.download_file(result.optimized_media, "optimized_video.mp4")
```

### Media Pipeline Integration

```python
from loop_optimizer import LoopOptimizer
from scene_validator import SceneValidator
from sound_scaffold import SoundScaffold

# Create a media pipeline
optimizer = LoopOptimizer()
validator = SceneValidator()
sound_enhancer = SoundScaffold()

# Process the media through the pipeline
validated_media = validator.validate("input.mp4")
looped_media = optimizer.process(validated_media)
final_media = sound_enhancer.enhance(looped_media)

print(f"Final media output: {final_media}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- FFmpeg project for media processing capabilities
- Google Cloud and Gemini API for cloud and AI integration
- PyTorch and TensorFlow communities for machine learning tools