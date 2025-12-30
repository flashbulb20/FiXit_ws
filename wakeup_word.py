import os
import numpy as np
import openwakeword
from openwakeword.model import Model
from scipy.signal import resample
from ament_index_python.packages import get_package_share_directory
from fixi_project import MicController

MODEL_NAME = "hello_rokey_8332_32.tflite"


class WakeupWord:
    def __init__(self, buffer_size):
        openwakeword.utils.download_models()
        package_share_dir = get_package_share_directory('fixi_project')    
        self.model_path = os.path.join(package_share_dir, 'models', MODEL_NAME)
        self.model = None
        self.model_name = MODEL_NAME.split(".", maxsplit=1)[0]
        self.stream = None
        self.buffer_size = buffer_size

    def is_wakeup(self):
        audio_chunk = np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False),
            dtype=np.int16,
        )
        audio_chunk = resample(audio_chunk, int(len(audio_chunk) * 16000 / 48000))
        outputs = self.model.predict(audio_chunk, threshold=0.1)
        confidence = outputs[self.model_name]
        print("confidence: ", confidence)
        # Wakeword 탐지
        if confidence > 0.1:
            print("Wakeword detected!")
            return True
        return False

    def set_stream(self, stream):
        self.model = Model(wakeword_models=[self.model_path])
        self.stream = stream


if __name__ == "__main__":
    Mic = MicController.MicController()
    Mic.open_stream()

    wakeup = WakeupWord(Mic.config.buffer_size)
    wakeup.set_stream(Mic.stream)
    while wakeup.is_wakeup() is False:
        pass
