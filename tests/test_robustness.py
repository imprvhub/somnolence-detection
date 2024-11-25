# tests/test_robustness.py
import pytest
import numpy as np
import cv2
import logging
from somnolence_detection import DrowsinessDetector

class TestRobustness:
    @pytest.fixture
    def detector(self):
        return DrowsinessDetector()

    def test_camera_failure(self, detector, mock_video_capture, mocker):
        logging.info("Testing camera failure handling")
        mocker.patch('cv2.VideoCapture', return_value=mock_video_capture)
        mock_video_capture.isOpened.return_value = False
        
        with pytest.raises(Exception, match="Could not open camera"):
            detector.start_detection()

    def test_frame_read_failure(self, detector, mock_video_capture, mocker):
        logging.info("Testing frame read failure")
        mock_video_capture.read.return_value = (False, None)
        mocker.patch('cv2.VideoCapture', return_value=mock_video_capture)
        
        detector.start_detection()
        mock_video_capture.release.assert_called_once()

    @pytest.mark.parametrize("resolution", [
        (320, 240, 3),
        (640, 480, 3),
        (1280, 720, 3)
    ])
    def test_different_resolutions(self, detector, resolution):
        logging.info(f"Testing resolution {resolution}")
        frame = np.zeros(resolution, dtype=np.uint8)
        processed_frame = detector.process_frame(frame)
        assert processed_frame.shape == resolution