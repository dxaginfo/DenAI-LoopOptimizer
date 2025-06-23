#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoopOptimizer API Server

This module provides a REST API for the LoopOptimizer tool.
"""

import os
import json
import uuid
import logging
import tempfile
from typing import Dict, Any, Optional
from dataclasses import asdict

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

from loop_optimizer import LoopOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure uploads
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mp3', 'wav', 'ogg', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB max upload

# Initialize optimizer
optimizer = LoopOptimizer()

# Helper functions
def allowed_file(filename: str) -> bool:
    """Check if a file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file) -> str:
    """Save an uploaded file and return the path."""
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    return filepath


# API routes
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': '0.1.0'
    })


@app.route('/api/v1/optimize', methods=['POST'])
def optimize():
    """
    Process a media file to create an optimized loop.
    
    Accepts either a file upload or a URL to a media file.
    """
    try:
        # Check for API key if configured
        api_key = os.environ.get('API_KEY')
        if api_key:
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer ') or auth_header[7:] != api_key:
                return jsonify({
                    'status': 'error',
                    'message': 'Unauthorized - Invalid or missing API key'
                }), 401
        
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400
            
            if not allowed_file(file.filename):
                return jsonify({
                    'status': 'error',
                    'message': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400
            
            media_source = save_uploaded_file(file)
        
        # Handle JSON payload with URL
        elif request.is_json:
            data = request.get_json()
            media_source = data.get('mediaSource')
            
            if not media_source:
                return jsonify({
                    'status': 'error',
                    'message': 'No mediaSource provided in request'
                }), 400
            
            # For URLs, download the file
            if media_source.startswith(('http://', 'https://')):
                import requests
                from urllib.parse import urlparse
                
                # Get filename from URL
                parsed_url = urlparse(media_source)
                filename = os.path.basename(parsed_url.path)
                if not filename or not allowed_file(filename):
                    filename = f"download.mp4"  # Default to MP4 if no extension
                
                # Download the file
                response = requests.get(media_source, stream=True)
                if response.status_code != 200:
                    return jsonify({
                        'status': 'error',
                        'message': f'Failed to download media from URL: {response.status_code}'
                    }), 400
                
                # Save to temp file
                unique_filename = f"{uuid.uuid4()}_{secure_filename(filename)}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                media_source = filepath
        else:
            return jsonify({
                'status': 'error',
                'message': 'No file or mediaSource provided'
            }), 400
        
        # Get processing parameters
        media_type = request.form.get('mediaType') or \
                    (request.json.get('mediaType') if request.is_json else None) or \
                    "video"
        
        crossfade_duration = float(request.form.get('crossfadeDuration') or \
                                  (request.json.get('crossfadeDuration') if request.is_json else None) or \
                                  0.5)
        
        optimization_level = request.form.get('optimizationLevel') or \
                            (request.json.get('optimizationLevel') if request.is_json else None) or \
                            "medium"
        
        output_format = request.form.get('outputFormat') or \
                       (request.json.get('outputFormat') if request.is_json else None)
        
        # Process the media
        result = optimizer.process(
            media_source=media_source,
            media_type=media_type,
            crossfade_duration=crossfade_duration,
            optimization_level=optimization_level,
            output_format=output_format
        )
        
        # Prepare response
        response_data = {
            'status': result.status,
            'processingTime': result.processing_time
        }
        
        if result.status == 'success':
            # Create file URL for download
            file_uuid = str(uuid.uuid4())
            filename = os.path.basename(result.optimized_media)
            
            # Store mapping for later retrieval
            app.config.setdefault('OUTPUT_FILES', {})[file_uuid] = result.optimized_media
            
            # Add file download URL to response
            download_url = f"/api/v1/download/{file_uuid}"
            response_data['optimizedMedia'] = download_url
            
            # Add loop points
            response_data['loopPoints'] = [asdict(lp) for lp in result.loop_points]
        else:
            response_data['errors'] = result.errors
            
        return jsonify(response_data)
        
    except Exception as e:
        logger.exception("Error processing optimization request")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/v1/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download a processed file."""
    try:
        output_files = app.config.get('OUTPUT_FILES', {})
        if file_id not in output_files:
            return jsonify({
                'status': 'error',
                'message': 'File not found'
            }), 404
        
        file_path = output_files[file_id]
        return send_file(file_path, as_attachment=True)
    
    except Exception as e:
        logger.exception("Error serving file")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Static HTML interface
@app.route('/', methods=['GET'])
def index():
    """Serve the simple HTML interface."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LoopOptimizer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            input, select {
                width: 100%;
                padding: 8px;
                box-sizing: border-box;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #2980b9;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: none;
            }
            .success {
                background-color: #d4edda;
                border-color: #c3e6cb;
            }
            .error {
                background-color: #f8d7da;
                border-color: #f5c6cb;
            }
            #loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <h1>LoopOptimizer</h1>
        <p>Upload a media file to create a seamlessly looping version.</p>
        
        <div class="form-group">
            <label for="mediaFile">Media File:</label>
            <input type="file" id="mediaFile" accept=".mp4,.avi,.mov,.mp3,.wav,.ogg,.webm">
        </div>
        
        <div class="form-group">
            <label for="mediaType">Media Type:</label>
            <select id="mediaType">
                <option value="video">Video</option>
                <option value="audio">Audio</option>
                <option value="both">Both</option>
            </select>
        </div>
        
        <div class="form-group">
            <label for="crossfade">Crossfade Duration (seconds):</label>
            <input type="number" id="crossfade" value="0.5" min="0.1" step="0.1">
        </div>
        
        <div class="form-group">
            <label for="optimization">Optimization Level:</label>
            <select id="optimization">
                <option value="low">Low</option>
                <option value="medium" selected>Medium</option>
                <option value="high">High</option>
            </select>
        </div>
        
        <button id="submitBtn">Create Loop</button>
        
        <div id="loading">
            Processing... This may take a few minutes for large files.
        </div>
        
        <div id="result"></div>
        
        <script>
            document.getElementById('submitBtn').addEventListener('click', async () => {
                const fileInput = document.getElementById('mediaFile');
                if (!fileInput.files.length) {
                    alert('Please select a file');
                    return;
                }
                
                const file = fileInput.files[0];
                const mediaType = document.getElementById('mediaType').value;
                const crossfade = document.getElementById('crossfade').value;
                const optimization = document.getElementById('optimization').value;
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('mediaType', mediaType);
                formData.append('crossfadeDuration', crossfade);
                formData.append('optimizationLevel', optimization);
                
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                loading.style.display = 'block';
                result.style.display = 'none';
                
                try {
                    const response = await fetch('/api/v1/optimize', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    
                    if (data.status === 'success') {
                        result.className = 'success';
                        result.innerHTML = `
                            <h3>Success!</h3>
                            <p>Processing time: ${data.processingTime.toFixed(2)} seconds</p>
                            <p>Loop points: Start=${data.loopPoints[0].start_time.toFixed(2)}s, 
                               End=${data.loopPoints[0].end_time.toFixed(2)}s</p>
                            <p>Quality score: ${data.loopPoints[0].quality_score.toFixed(3)}</p>
                            <p><a href="${data.optimizedMedia}" download>Download optimized media</a></p>
                        `;
                    } else {
                        result.className = 'error';
                        result.innerHTML = `
                            <h3>Error</h3>
                            <p>${data.message || data.errors[0].message}</p>
                        `;
                    }
                } catch (error) {
                    loading.style.display = 'none';
                    result.style.display = 'block';
                    result.className = 'error';
                    result.innerHTML = `
                        <h3>Error</h3>
                        <p>An unexpected error occurred: ${error.message}</p>
                    `;
                }
            });
        </script>
    </body>
    </html>
    """
    return html


if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Start the server
    app.run(host='0.0.0.0', port=port, debug=False)