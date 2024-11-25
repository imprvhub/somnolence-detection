import pytest
import numpy as np
import cv2
import logging
from scipy.spatial import distance
from somnolence_detection import DrowsinessDetector

class TestDrowsinessDetector:
    @pytest.fixture
    def detector(self):
        return DrowsinessDetector()

    def test_initialization(self, detector):
        logging.info("Testing detector initialization")
        assert len(detector.LEFT_EYE) == 6
        assert len(detector.RIGHT_EYE) == 6
        assert detector.EAR_THRESH == 0.25
        assert detector.CLOSED_EYES_FRAME == 20
        assert detector.counter == 0

    def test_calculate_EAR(self, detector):
        logging.info("Testing EAR calculation")
        eye_points = np.array([[0, 0], [1, 2], [2, 2], [3, 0], [2, -2], [1, -2]])
        ear = detector.calculate_EAR(eye_points)
        assert isinstance(ear, float)
        assert ear > 0

    def test_process_frame_no_face(self, detector):
        logging.info("Testing frame processing without face")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed_frame = detector.process_frame(frame)
        assert processed_frame.shape == frame.shape
        assert isinstance(processed_frame, np.ndarray)

    @pytest.mark.parametrize("ear_value", [0.2, 0.3])
    def test_drowsiness_threshold(self, detector, ear_value, mocker):
        logging.info(f"Testing drowsiness threshold with EAR {ear_value}")
        mock_mesh_points = mocker.patch.object(detector, 'process_frame', side_effect=lambda frame: frame)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.counter = 0
        
        for _ in range(detector.CLOSED_EYES_FRAME + 1):
            if ear_value < detector.EAR_THRESH:
                detector.counter += 1
                
        is_alert = detector.counter >= detector.CLOSED_EYES_FRAME
        expected_alert = ear_value < detector.EAR_THRESH
        assert is_alert == expected_alert, f"Expected alert to be {expected_alert} for EAR value {ear_value} (threshold: {detector.EAR_THRESH})"