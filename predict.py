# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import cv2
import subprocess
import numpy as np
from PIL import Image
from transparent_background import Remover

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        video: Path = Input(description="Grayscale input image"),
        mode: str = Input(description="Mode of operation", default="Normal", choices=["Fast", "Normal"])
    ) -> Path:
        """Run a single prediction on the model"""
        if mode == 'Fast':
            remover = Remover(mode='fast')
        else:
            remover = Remover()

        input_video = str(video)
        cap = cv2.VideoCapture(input_video)
        writer = None
        tmpname = "/tmp/tmp.mp4"
        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).convert('RGB')

            if writer is None:
                writer = cv2.VideoWriter(str(tmpname), cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), img.size)

            processed_frames += 1
            print(f"Processing frame {processed_frames}")
            out = remover.process(img, type='green')
            writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))

        cap.release()
        writer.release()
        output_path = "/tmp/output.mp4"
        # ffmpeg command to add codec libx264 to the video
        subprocess.run(["ffmpeg", "-i", tmpname, "-c:v", "libx264", "-crf", "0", output_path], check=True)
        return Path(output_path)
