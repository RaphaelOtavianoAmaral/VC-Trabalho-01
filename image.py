import cv2 as cv #Import OpenCV
import numpy as np #Import NumPy
import matplotlib.pyplot as plt #Import pyplot


class Image():
  'Class for image processing'

  _loaded: bool = False

  def __init__(self, image_path: str = None, image_array: np.ndarray = None, color_read: str = None):
    self.image_path = image_path
    self.image_array = image_array
    self.color_read = color_read
    self._load_image()

  def get_image_array(self):
    return self.image_array

  def get_image_path(self):
    return self.image_path

  def set_image_path(self, image_path: str):
    self.image_path = image_path

  def set_image_array(self, image_array: np.ndarray):
    self.image_array = image_array

  def _load_image(self):
    if not self._loaded and self.image_path != None:
      if self.color_read is None:
        self.cv_image = cv.imread(self.image_path)
      else:
        self.cv_image = cv.imread(self.image_path,cv.IMREAD_GRAYSCALE)
      self.image_array = np.array(self.cv_image)
      self.shape = self.image_array.shape
      self._loaded = True
    else:
      self.shape = self.image_array.shape


  def save_image(self, path: str = ""):
    if self.image_array != None:
      cv.imwrite(path, self.image_array)

  def show_image(self):
    cv.imshow('Image', self.image_array)
    cv.waitKey(0)
    cv.destroyAllWindows()

  def plot_image(self):
    img_rbg = cv.cvtColor(self.image_array, cv.COLOR_BGR2RGB)
    plt.imshow(img_rbg)
    plt.show()
