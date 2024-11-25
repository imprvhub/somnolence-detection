# tests/conftest.py
import os
import sys
import pytest
import numpy as np
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log_directory = "test_logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_filename = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_directory, log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

@pytest.fixture(autouse=True)
def log_test_info(request):
    logging.info(f"Starting test: {request.node.name}")
    yield
    logging.info(f"Finished test: {request.node.name}")

@pytest.fixture
def mock_video_capture(mocker):
    mock_cap = mocker.Mock()
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.isOpened.return_value = True
    mock_cap.release = mocker.Mock()
    return mock_cap