# vision-robot-navigation-aruco
Real-time robot navigation using ArUco markers, OpenCV, and A* path planning

This project is implemented as a single integrated script (marker.py) that combines multiple components of a robotics pipeline:
	•	ArUco Marker Detection using OpenCV for real-time environment perception
	•	Path Planning (A*) for computing optimal navigation paths
	•	Navigation Decision Logic to determine movement directions (LEFT / RIGHT / FORWARD)

Design Approach
The system is designed as a self-contained prototype, where all modules interact within a single script for:
	•	Faster development and testing
	•	Seamless integration of vision and control
	•	Real-time performance without inter-file dependencies

Output Visualization
Below is a sample output showing real-time detection and path planning:
    •	Green boundary: Arena detection
	•	Red box: Obstacle
	•	Yellow path: Planned path (A*)
	•	BOT: Robot position
	•	GOAL: Target location

Future Improvements
	•	Modularizing into separate files (marker.py, path_planning.py, etc.)
	•	Adding obstacle avoidance with dynamic updates
	•	Integrating with physical robot hardware
