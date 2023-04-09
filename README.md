# Garage Door State Detection and MQTT Publishing

This project detects the state of a garage door (open or closed) using a video stream and publishes the state to an MQTT topic. The state detection is performed using a K-Nearest Neighbors (KNN) classifier trained on a set of sample images. The project is composed of two main components: video processing using OpenCV and MQTT communication using Paho MQTT library.

## Prerequisites

- Python 3.7 or higher
- OpenCV
- Paho MQTT
- scikit-learn

To install the required libraries, run the following command:

```bash
pip install opencv-python opencv-python-headless paho-mqtt scikit-learn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/erichensley/garage-door-state-detection.git
cd garage-door-state-detection
```

2. Update the `video_stream` variable in `server.py` with the RTSP URL of your camera.

3. Update the MQTT settings in `server.py` with your MQTT server's IP address, queue, username, and password.

4. Run the `server.py` script:

```bash
python server.py
```

The script will continuously monitor the garage door's state using the video stream and publish the state to the specified MQTT topic.

## Project Structure

- `server.py`: Main script that processes the video stream, detects the garage door state, and publishes it to an MQTT topic.
- `mqtt.py`: An example script for displaying a video stream from the camera.
- `save_snapshot.py`: Helper file for creating images from the camera for training
- `train_ai_model.py`: Helper for training the ML model to detect open/closed state

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Created by Eric Hensley. Inspired by [gas_monitor].(https://github.com/erkexzcx/gas_monitor/blob/main/instructions/part_1_preparation.md)
