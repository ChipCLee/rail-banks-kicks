import io
import unittest

from PIL import Image

from image_input import InvalidImageError, decode_camera_image


class TestCameraImageDecoding(unittest.TestCase):
    def test_applies_iphone_exif_orientation(self):
        image = Image.new("RGB", (40, 20), "blue")
        exif = Image.Exif()
        exif[274] = 6  # Rotate 90 degrees clockwise for display.
        content = io.BytesIO()
        image.save(content, format="JPEG", exif=exif)

        decoded = decode_camera_image(content.getvalue())
        self.assertEqual(decoded.shape, (40, 20, 3))

    def test_bounds_large_camera_dimension(self):
        image = Image.new("RGB", (200, 100), "green")
        content = io.BytesIO()
        image.save(content, format="PNG")

        decoded = decode_camera_image(content.getvalue(), max_dimension=50)
        self.assertEqual(decoded.shape, (25, 50, 3))

    def test_rejects_pixel_bomb_before_full_decode(self):
        image = Image.new("RGB", (100, 100), "red")
        content = io.BytesIO()
        image.save(content, format="PNG")

        with self.assertRaises(InvalidImageError):
            decode_camera_image(content.getvalue(), max_pixels=9_999)

    def test_rejects_empty_upload(self):
        with self.assertRaises(InvalidImageError):
            decode_camera_image(b"")


if __name__ == "__main__":
    unittest.main()
