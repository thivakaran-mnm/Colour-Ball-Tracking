# Ball Tracking and Quadrant Detection System

## Overview

This project focuses on detecting and tracking colored balls within defined quadrants in a video. The system identifies and logs the entry and exit of balls in these quadrants using computer vision techniques.

## Table of Contents
- [Overview](#overview)
- [Dataset Description](#dataset-description)
- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Usage](#usage)
- [Function Descriptions](#function-descriptions)
- [Conclusion](#conclusion)

## Dataset Description

The project utilizes a video input containing colored balls (yellow, white, green, and orange) moving within a frame divided into four quadrants by red lines. The video captures the entry and exit of these balls into and out of the quadrants.

## Pre-requisites

- Python 3.6 
- OpenCV
- NumPy
- Pandas

## Installation

Clone the repository:

   git clone https://github.com/thivakaran_mnm/Colour-Ball-Tracking.git
   
## Usage

- Prepare the dataset by placing the video files in the appropriate directory.
- Run the main script to process the video and detect ball movements.

## Function Descriptions

- **rgb_to_hsv(r, g, b):** Converts RGB color values to HSV color space.
- **ball_colors:** Dictionary containing HSV color ranges for each ball color.
- **get_quadrant(x, y, vertical_line, horizontal_line):** Determines the quadrant in which a ball is located based on its coordinates.
- **detect_balls(frame, lower_color, upper_color):** Detects balls in a given frame based on color range.
- **draw_quadrants(frame, vertical_line, horizontal_line):** Draws quadrant lines on the frame.
- **Main Processing Loop:** Captures frames from the video, detects balls, tracks their movements, logs entry and exit events, and saves the processed video and events.

## Conclusion
This project successfully tracks the movement of colored balls within defined quadrants in a video. It identifies and logs entry and exit events for each ball in real-time using computer vision techniques. The system can be extended or improved by incorporating more sophisticated object detection models or handling additional colors and shapes.
