from PIL import Image
import PIL.ImageOps

class RgbConverter:

    internalMode = "RGB"

    @staticmethod
    def apply(img):
        return img

    @staticmethod
    def unapply(img):
        return img

class BwNaiveConverter:

    internalMode = "RGB"

    @staticmethod
    def apply(img):
        rgb2bw = (
            0.33, 0.33, 0.33, 0,
            0.33, 0.33, 0.33, 0,
            0.33, 0.33, 0.33, 0 )
        return img.convert("RGB", rgb2bw)

    @staticmethod
    def unapply(img):
        return img

class BwLumaConverter:

    internalMode = "RGB"

    @staticmethod
    def apply(img):
        rgb2bw = (
            0.299, 0.587, 0.114, 0,
            0.299, 0.587, 0.114, 0,
            0.299, 0.587, 0.114, 0 )
        return img.convert("RGB", rgb2bw)

    @staticmethod
    def unapply(img):
        return img

class XyzConverter:

    internalMode = "RGB"

    @staticmethod
    def apply(img):
        rgb2xyz = (
            0.412453, 0.357580, 0.180423, 0,
            0.212671, 0.715160, 0.072169, 0,
            0.019334, 0.119193, 0.950227, 0 )
        return img.convert("RGB", rgb2xyz)

    @staticmethod
    def unapply(img):
        xyz2rgb = (
            3.24048, -1.53715, -0.498536, 0,
            -0.969255, 1.87599, 0.0415559, 0,
            0.0556466, 0.204041, 1.05731, 0 )
        return img.convert("RGB", xyz2rgb)

