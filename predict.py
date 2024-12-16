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
        
    def hex_to_rgb(self, hex_color: str) -> list:
        """Convert hex color string to RGB tuple"""
        # Remove '#' if present
        hex_color = hex_color.lstrip('#')
        # Convert hex to RGB
        return list(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def predict(
        self,
        video: Path = Input(description="Grayscale input image"),
        mode: str = Input(description="Mode of operation", default="Normal", choices=["Fast", "Normal"]),
        background_color: str = Input(
            description="Background color in hex format (e.g., '#FFFFFF' for white)",
            default="#FFFFFF"
        )
    ) -> Path:
        """Run a single prediction on the model"""
        if mode == 'Fast':
            remover = Remover(mode='fast')
        else:
            remover = Remover()

        # Convert hex color to RGB
        try:
            bg_color = str(self.hex_to_rgb(background_color))
            print(f"Background color: {bg_color}")
        except ValueError:
            raise ValueError("Invalid hex color format. Please use format '#RRGGBB'")

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
            # Update to use custom background color
            out = remover.process(img, type=bg_color)
            writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))

        cap.release()
        writer.release()
        output_path = "/tmp/output.mp4"
        # ffmpeg command to add codec libx264 to the video
        subprocess.run(["ffmpeg", "-y", "-i", tmpname, "-c:v", "libx264", "-crf", "0", output_path], check=True)
        return Path(output_path)
