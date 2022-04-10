import json
import os
import numpy as np

FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'cameras.json')

class Camera():

    def __init__(self, name, focal_length, sensor_size,
                 pixels_along_track, pixels_across_track) -> None:
        self.name = name
        self.focal_length = focal_length
        self.sensor_size = sensor_size
        self.pixels_along_track = pixels_along_track
        self.pixels_across_track = pixels_across_track


    def save(self):
        """Save new camera in cameras.json file."""
        with open(FILE_PATH, "r") as cameras_file:
            data = json.load(cameras_file)

            is_duplicate = False
            for i, camera in enumerate(data):
                if self.name == camera['name']:
                    data[i] = self.__dict__
                    is_duplicate = True

            if not is_duplicate:
                data.append(self.__dict__)
        with open(FILE_PATH, "w") as cameras_file:
            json.dump(data, cameras_file, indent=4)


    def delete(self):
        """Delete selected camera from cameras.json file."""
        with open(FILE_PATH, "r") as cameras_file:
            data = json.load(cameras_file)

        data.remove(self.__dict__)

        with open(FILE_PATH, "w") as cameras_file:
            json.dump(data, cameras_file, indent=4)


    def image_corners(self):
        """Return array of x, y, z coordinates of image corners in image space."""
        x = (self.sensor_size) * self.pixels_along_track / 2
        y = (self.sensor_size) * self.pixels_across_track / 2
        image_corners_coordinates = np.array([[-x, y, -self.focal_length],
                                            [-x, -y, -self.focal_length],
                                            [x, -y, -self.focal_length],
                                            [x, y, -self.focal_length]
                                            ])
        return image_corners_coordinates
