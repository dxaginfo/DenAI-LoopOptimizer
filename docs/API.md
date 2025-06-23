# LoopOptimizer API Reference

This document provides comprehensive information about the LoopOptimizer REST API endpoints, request formats, and response structures.

## Base URL

When running locally, the base URL is:
```
http://localhost:5000
```

When deployed to Google Cloud Run, the base URL will be:
```
https://loop-optimizer-[PROJECT_ID].run.app
```

## Authentication

The API uses Bearer token authentication. Include the following header in your requests:

```
Authorization: Bearer YOUR_API_KEY
```

Authentication can be configured by setting the `API_KEY` environment variable when starting the server.

## Endpoints

### Health Check

Verify that the API is running and operational.

**Endpoint:** `GET /api/v1/health`

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### Optimize Media

Process a media file to create a seamlessly looping version.

**Endpoint:** `POST /api/v1/optimize`

**Request:**

The API supports two methods of providing media:

1. **File Upload** (multipart/form-data):
   ```
   Content-Type: multipart/form-data
   
   file: [BINARY FILE DATA]
   mediaType: "video" | "audio" | "both"
   crossfadeDuration: 0.5
   optimizationLevel: "low" | "medium" | "high"
   outputFormat: "mp4" | "mp3" | etc. (optional)
   ```

2. **JSON Request** (application/json):
   ```json
   {
     "mediaSource": "https://example.com/video.mp4",
     "mediaType": "video",
     "crossfadeDuration": 0.5,
     "optimizationLevel": "medium",
     "outputFormat": "mp4"
   }
   ```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file / mediaSource | File / String | Yes | The media file to process, either as a file upload or URL |
| mediaType | String | No | Type of media: "audio", "video", or "both" (default: "video") |
| crossfadeDuration | Number | No | Duration of crossfade in seconds (default: 0.5) |
| optimizationLevel | String | No | Level of optimization: "low", "medium", or "high" (default: "medium") |
| outputFormat | String | No | Output format (e.g., "mp4", "mp3") - if not specified, uses same format as input |

**Successful Response:**
```json
{
  "status": "success",
  "processingTime": 12.34,
  "optimizedMedia": "/api/v1/download/550e8400-e29b-41d4-a716-446655440000",
  "loopPoints": [
    {
      "start_time": 0.0,
      "end_time": 15.23,
      "quality_score": 0.85,
      "notes": "Created with medium optimization"
    }
  ]
}
```

**Error Response:**
```json
{
  "status": "error",
  "errors": [
    {
      "code": "file_not_found",
      "message": "File not found: /path/to/file.mp4"
    }
  ],
  "processingTime": 0.15
}
```

### Download Optimized Media

Download the optimized media file generated from a successful optimization request.

**Endpoint:** `GET /api/v1/download/{file_id}`

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| file_id | String (path) | Yes | The file ID returned in the optimization response |

**Response:**

The binary file content with appropriate Content-Type and Content-Disposition headers.

## Error Codes

| Code | Description |
|------|-------------|
| file_not_found | The specified file could not be found |
| no_loop_points | No suitable loop points were found in the media |
| processing_error | An error occurred during processing |
| invalid_format | The media format is not supported |

## Usage Examples

### cURL

```bash
# Upload a file
curl -X POST http://localhost:5000/api/v1/optimize \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@/path/to/video.mp4" \
  -F "mediaType=video" \
  -F "optimizationLevel=high"

# Process a URL
curl -X POST http://localhost:5000/api/v1/optimize \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "mediaSource": "https://example.com/video.mp4",
    "mediaType": "video",
    "optimizationLevel": "high"
  }'
```

### Python

```python
import requests

# Upload a file
files = {'file': open('video.mp4', 'rb')}
data = {
    'mediaType': 'video',
    'optimizationLevel': 'high'
}
response = requests.post(
    'http://localhost:5000/api/v1/optimize',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    files=files,
    data=data
)
result = response.json()

# Process a URL
response = requests.post(
    'http://localhost:5000/api/v1/optimize',
    headers={
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json'
    },
    json={
        'mediaSource': 'https://example.com/video.mp4',
        'mediaType': 'video',
        'optimizationLevel': 'high'
    }
)
result = response.json()

# Download the optimized file
if result['status'] == 'success':
    download_url = f"http://localhost:5000{result['optimizedMedia']}"
    download_response = requests.get(download_url)
    with open('optimized_video.mp4', 'wb') as f:
        f.write(download_response.content)
```

## Limitations

- Maximum file size: 200MB
- Supported file formats: mp4, avi, mov, mp3, wav, ogg, webm
- API requests have a 10-minute timeout for processing large files

## Rate Limiting

The API implements basic rate limiting to prevent abuse. By default, clients are limited to 100 requests per hour per API key or IP address.

## Versioning

The API follows semantic versioning. The current version is v1. Future breaking changes will be introduced in new major versions.