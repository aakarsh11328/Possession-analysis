# AI-Powered Football Video Analysis

## Overview
An AI-powered football analysis system for player detection,
tracking, team classification and ball possession estimation.

## Features
- Player detection using YOLO
- Player tracking using ByteTrack
- Jersey-colour-based team classification
- Goalkeeper identification
- Ball possession estimation (H/A/D/X)
- Frame-by-frame CSV generation

## Technologies Used
- Python
- OpenCV
- YOLO
- ByteTrack
- NumPy

## Installation
pip install ultralytics opencv-python numpy

## How to Run
python3 possession_analysis.py

## Output
Generates a CSV containing frame-by-frame possession data
and match statistics.

## Limitations
Possession is estimated using player and ball positions.
Results may contain occasional classification errors.
