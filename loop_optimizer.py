#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoopOptimizer Core Module

This module provides the core functionality for identifying and optimizing
loop points in audio and video media.
"""

import os
import json
import logging
import tempfile
import argparse
import subprocess
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

# Optional imports - fail gracefully if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoopPoint:
    """Data class for storing loop point information."""
    start_time: float
    end_time: float
    quality_score: float
    notes: Optional[str] = None


@dataclass
class ProcessingResult:
    """Data class for storing media processing results."""
    status: str
    optimized_media: Optional[str] = None
    loop_points: List[LoopPoint] = field(default_factory=list)
    processing_time: float = 0.0
    errors: List[Dict[str, str]] = field(default_factory=list)


class LoopOptimizer:
    """
    Core class for detecting and optimizing loop points in media files.
    
    This class provides methods to analyze audio and video files,
    identify optimal loop points, and generate seamlessly looping media.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LoopOptimizer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._check_dependencies()
        self._initialize_gemini()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available."""
        # Check for FFmpeg
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            if result.returncode != 0:
                logger.warning("FFmpeg not found. Media processing will fail.")
        except FileNotFoundError:
            logger.warning("FFmpeg not found. Media processing will fail.")
        
        # Check for PyTorch/TensorFlow
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Using basic processing methods.")
    
    def _initialize_gemini(self) -> None:
        """Initialize Gemini API if available."""
        if GEMINI_AVAILABLE:
            # Check for API key in config or environment
            api_key = self.config.get('gemini_api_key') or os.environ.get('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                logger.info("Gemini API initialized successfully")
            else:
                logger.warning("Gemini API key not found. Enhanced analysis unavailable.")
        else:
            logger.warning("Gemini API not available. Enhanced analysis unavailable.")
    
    def _extract_audio(self, media_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio data from media file.
        
        Args:
            media_path: Path to the media file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        # Create temporary file for audio extraction
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio_path = temp_audio.name
        
        # Extract audio using FFmpeg
        cmd = [
            "ffmpeg", "-i", media_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            temp_audio_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Load audio data
            audio_data = np.memmap(temp_audio_path, dtype='int16', mode='r')
            audio_data = audio_data.reshape(-1, 2)
            
            # Remove temporary file
            os.unlink(temp_audio_path)
            
            return audio_data, 44100
        except subprocess.CalledProcessError as e:
            logger.error(f"Error extracting audio: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)
            raise RuntimeError(f"Failed to extract audio from {media_path}")
    
    def _analyze_audio_similarity(self, audio_data: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Analyze audio for potential loop points based on similarity.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            List of tuples (start_time, end_time, similarity_score)
        """
        # Convert to mono if stereo
        if audio_data.ndim > 1:
            audio_mono = audio_data.mean(axis=1)
        else:
            audio_mono = audio_data
        
        # Normalize
        audio_mono = audio_mono / np.max(np.abs(audio_mono))
        
        # Parameters
        sample_rate = 44100
        min_loop_duration = int(1.0 * sample_rate)  # 1 second minimum
        max_loop_duration = int(30.0 * sample_rate)  # 30 seconds maximum
        
        # Analysis results
        loop_candidates = []
        
        # Simple autocorrelation approach
        for offset in range(min_loop_duration, min(max_loop_duration, len(audio_mono) // 2), sample_rate):
            # Compare first half with second half shifted by offset
            segment_length = min(offset, len(audio_mono) - offset)
            first_segment = audio_mono[:segment_length]
            second_segment = audio_mono[offset:offset + segment_length]
            
            # Calculate correlation
            correlation = np.corrcoef(first_segment, second_segment)[0, 1]
            
            # Calculate RMS difference
            rms_diff = np.sqrt(np.mean((first_segment - second_segment) ** 2))
            
            # Combined score (higher is better)
            similarity_score = max(0, (correlation * 0.7) - (rms_diff * 0.3))
            
            if similarity_score > 0.6:  # Reasonable threshold
                start_time = 0
                end_time = offset / sample_rate
                loop_candidates.append((start_time, end_time, similarity_score))
        
        # Sort by score (highest first)
        loop_candidates.sort(key=lambda x: x[2], reverse=True)
        
        return loop_candidates[:5]  # Return top 5 candidates
    
    def _create_loop(self, media_path: str, output_path: str, 
                    start_time: float, end_time: float,
                    crossfade_duration: float = 0.5) -> str:
        """
        Create a looping media file with crossfade.
        
        Args:
            media_path: Path to the input media file
            output_path: Path for the output media file
            start_time: Loop start time in seconds
            end_time: Loop end time in seconds
            crossfade_duration: Duration of crossfade in seconds
            
        Returns:
            Path to the created looping media file
        """
        # Create the FFmpeg command for loop creation with crossfade
        cmd = [
            "ffmpeg", 
            "-i", media_path,
            "-filter_complex",
            f"[0:v]trim=start={start_time}:end={end_time},setpts=PTS-STARTPTS[v1];"
            f"[0:a]atrim=start={start_time}:end={end_time},asetpts=PTS-STARTPTS[a1];"
            f"[v1][v1]xfade=transition=fade:duration={crossfade_duration}:offset={end_time-start_time-crossfade_duration}[vout];"
            f"[a1][a1]acrossfade=d={crossfade_duration}:c1=tri:c2=tri[aout]",
            "-map", "[vout]", "-map", "[aout]",
            "-shortest", output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating loop: {e}")
            raise RuntimeError(f"Failed to create looping media at {output_path}")
    
    def _enhance_with_gemini(self, media_path: str, loop_candidates: List[Tuple[float, float, float]]) -> List[Dict[str, Any]]:
        """
        Use Gemini API to enhance loop point selection if available.
        
        Args:
            media_path: Path to the media file
            loop_candidates: List of candidate loop points
            
        Returns:
            Enhanced analysis of loop points
        """
        if not GEMINI_AVAILABLE:
            return []
        
        try:
            # Get media info
            cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", 
                   "-show_format", "-show_streams", media_path]
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            media_info = json.loads(result.stdout)
            
            # Format candidates
            candidates_text = "\n".join([
                f"Candidate {i+1}: Start={c[0]:.2f}s, End={c[1]:.2f}s, Score={c[2]:.3f}"
                for i, c in enumerate(loop_candidates)
            ])
            
            # Prepare prompt for Gemini
            prompt = f"""
            Analyze these potential loop points for a media file with the following characteristics:
            
            Media information:
            {json.dumps(media_info, indent=2)}
            
            Loop candidates:
            {candidates_text}
            
            For each candidate, provide:
            1. A quality assessment (1-10)
            2. Suggested improvements
            3. Potential issues with the loop
            4. Recommended crossfade duration
            
            Format your response as JSON with fields: quality_assessment, suggested_improvements, 
            potential_issues, and recommended_crossfade.
            """
            
            # Call Gemini API
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            # Extract JSON from response
            result_text = response.text
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                logger.warning("Could not extract JSON from Gemini response")
                return []
                
        except Exception as e:
            logger.error(f"Error using Gemini API: {e}")
            return []
    
    def process(self, 
                media_source: str,
                media_type: str = "video",
                target_duration: Optional[float] = None,
                crossfade_duration: float = 0.5,
                optimization_level: str = "medium",
                output_format: Optional[str] = None) -> ProcessingResult:
        """
        Process a media file to create an optimized loop.
        
        Args:
            media_source: Path or URL to the media file
            media_type: Type of media ("audio", "video", or "both")
            target_duration: Target duration for the loop in seconds
            crossfade_duration: Duration of crossfade in seconds
            optimization_level: Level of optimization ("low", "medium", "high")
            output_format: Output format (e.g., "mp4", "mp3")
            
        Returns:
            ProcessingResult object with results and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Validate inputs
            if not os.path.exists(media_source):
                return ProcessingResult(
                    status="error",
                    errors=[{"code": "file_not_found", "message": f"File not found: {media_source}"}],
                    processing_time=time.time() - start_time
                )
            
            # Extract audio for analysis
            audio_data, sample_rate = self._extract_audio(media_source)
            
            # Analyze for loop points
            loop_candidates = self._analyze_audio_similarity(audio_data)
            
            if not loop_candidates:
                return ProcessingResult(
                    status="error",
                    errors=[{"code": "no_loop_points", "message": "No suitable loop points found"}],
                    processing_time=time.time() - start_time
                )
            
            # Determine output format if not specified
            if not output_format:
                output_format = os.path.splitext(media_source)[1][1:]
            
            # Create output path
            base_name = os.path.splitext(os.path.basename(media_source))[0]
            output_path = f"{base_name}_looped.{output_format}"
            
            # Create the loop
            best_candidate = loop_candidates[0]  # Use highest scored candidate
            start_time_sec, end_time_sec, quality_score = best_candidate
            
            # Apply Gemini enhancement if available
            gemini_analysis = self._enhance_with_gemini(media_source, loop_candidates)
            
            # Create the optimized loop
            optimized_path = self._create_loop(
                media_source, 
                output_path, 
                start_time_sec, 
                end_time_sec, 
                crossfade_duration
            )
            
            # Create loop point object
            loop_point = LoopPoint(
                start_time=start_time_sec,
                end_time=end_time_sec,
                quality_score=quality_score,
                notes=f"Created with {optimization_level} optimization"
            )
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                status="success",
                optimized_media=optimized_path,
                loop_points=[loop_point],
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing media: {e}")
            return ProcessingResult(
                status="error",
                errors=[{"code": "processing_error", "message": str(e)}],
                processing_time=time.time() - start_time
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LoopOptimizer - Create seamless media loops")
    
    parser.add_argument("--input", required=True, help="Input media file path")
    parser.add_argument("--output", help="Output media file path")
    parser.add_argument("--media-type", choices=["audio", "video", "both"], 
                        default="video", help="Type of media to process")
    parser.add_argument("--optimization", choices=["low", "medium", "high"],
                       default="medium", help="Optimization level")
    parser.add_argument("--crossfade", type=float, default=0.5,
                       help="Crossfade duration in seconds")
    parser.add_argument("--gemini-api-key", help="Google Gemini API key")
    
    return parser.parse_args()


def main():
    """Main function for command line usage."""
    args = parse_args()
    
    # Configure optimizer
    config = {
        "gemini_api_key": args.gemini_api_key,
        "optimization_level": args.optimization
    }
    
    optimizer = LoopOptimizer(config)
    
    # Process the media
    output_path = args.output or f"{os.path.splitext(args.input)[0]}_looped{os.path.splitext(args.input)[1]}"
    
    print(f"Processing {args.input}...")
    result = optimizer.process(
        media_source=args.input,
        media_type=args.media_type,
        crossfade_duration=args.crossfade,
        optimization_level=args.optimization,
        output_format=os.path.splitext(output_path)[1][1:]
    )
    
    # Print results
    if result.status == "success":
        print(f"\nSuccess! Created optimized loop at: {result.optimized_media}")
        print(f"Loop points: Start={result.loop_points[0].start_time:.2f}s, End={result.loop_points[0].end_time:.2f}s")
        print(f"Quality score: {result.loop_points[0].quality_score:.3f}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
    else:
        print("\nError processing media:")
        for error in result.errors:
            print(f"  - {error['message']}")


if __name__ == "__main__":
    main()