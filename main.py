from image import Image
from filters import MeanFilter, MedianFilter, GaussianFilter, LaplacianFilter, PrewitFilter, SobelFilter


def run():
  img = Image("./images/lena.tif")
  print(img.shape)
  print(type(img.shape))
  print(len(img.shape))
  #img.plot_image()


def run_mean_filter():
  img = Image("./images/lena.tif")
  img.plot_image()
  
  filter = MeanFilter()

  filter_result = filter.operate_filter(img,5)

  filter_result.plot_image()


def run_median_filter():
  img = Image("./images/lena.tif")
  img.plot_image()
  
  filter = MedianFilter()

  filter_result = filter.operate_filter(img,5)

  filter_result.plot_image()


def run_gaussian_filter():
  img = Image("./images/lena.tif")
  img.plot_image()
  print("Original",img.image_array[:,:,0])
  
  filter = GaussianFilter()

  filter_result = filter.operate_filter(img,5)

  filter_result.plot_image()
  print("Filtrado",filter_result.image_array[:,:,0])


def run_laplacian_filter():
  img = Image("./images/lena.tif",color_read="Gray")
  img.plot_image()
  #print("Original",img.image_array[:,:,0])
  
  filter = LaplacianFilter(True,1)

  filter_result = filter.operate_filter(img,7)

  filter_result.plot_image()
  #print("Filtrado",filter_result.image_array[:,:,0])

def run_prewit_filter():
  img = Image("./images/lena.tif",color_read=None)
  img.plot_image()
  #print("Original",img.image_array[:,:,0])
  
  filter = PrewitFilter()

  filter_result = filter.operate_filter(img,2)

  filter_result.plot_image()
  #print("Filtrado",filter_result.image_array[:,:,0])

def run_sobel_filter():
  img = Image("./images/lena.tif",color_read="Gray")
  img.plot_image()
  #print("Original",img.image_array[:,:,0])

  filter = SobelFilter()

  filter_result = filter.operate_filter(img,3)

  filter_result.plot_image()
  #print("Filtrado",filter_result.image_array[:,:,0])


if __name__ == "__main__":
  run_sobel_filter()
