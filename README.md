
# Climber Hold Detection and Analysis

This project utilizes computer vision techniques to detect and analyze holds on a climbing wall using video input. The program tracks climber's movements, identifies stable holds, and calculates the center of gravity to predict potential next moves.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python installed on your machine along with the following packages:

- OpenCV
- Mediapipe
- NumPy

You can install all required packages using:

```
pip install -r requirements.txt
```

Setting Up
	1.	Clone the Repository:

git clone https://github.com/your-username/climber-hold-detection.git
cd climber-hold-detection


2.	Install Dependencies:
```
pip install -r requirements.txt
```

3.	Video Setup:
Place your climbing video in the project directory or specify the path to the video in the code:

cap = cv2.VideoCapture('path_to_your_video.mp4')


4.	Variable Configuration:
Adjust the variables in the code as needed:
	•	BUFFER_SIZE: Adjusts the buffer size for position tracking.
	•	MOVEMENT_THRESHOLD: Sets the threshold for movement sensitivity.
	•	board_angle_deg: Sets the board angle in degrees to adjust the calculations based on the wall’s inclination.

Running the Program

Run the script using Python:

python main.py

Features
	•	Hold Detection: Identifies stable positions where the climber’s limbs engage with holds.
	•	Center of Gravity Calculation: Dynamically calculates the center of gravity to predict climber’s movements.
	•	Tension Direction: Analyzes the direction of tension based on limb positioning and gravity.

Contributing

Contributions are welcome! Please feel free to submit a pull request.

License

This project is licensed under the MIT License - see the LICENSE.md file for details.
