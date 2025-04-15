from abc import ABC, abstractmethod
from image import Image
import numpy as np
import math

class FilterTemplate(ABC):
    'Abstract class for any filter'

    @abstractmethod
    def _create_kernel(self, size: int) -> None:
        pass

    @abstractmethod
    def _adapt_array(self, image_array: np.ndarray, kernel_size: int) -> None:
        pass

    @abstractmethod
    def operate_filter(self, image: Image, kernel_size: int) -> Image:
        pass

    

class MeanFilter(FilterTemplate):
    'Mean Filter for Image processing'

    def __init__(self) -> None:
        pass

    def _create_kernel(self, size) -> None:
        self.kernel = np.ones((size,size))

    def _adapt_array(self, image_array: np.ndarray, shape: tuple ,kernel_size: int) -> None:
        center = math.floor(kernel_size/2)

        if len(shape) == 2:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2))
            self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center)] = image_array
        elif len(shape) == 3:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2, shape[2]))

            for i in range(0, shape[2]):
                self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center), i] = image_array[:,:,i]

    
    def operate_filter(self, image: Image, kernel_size: int = 2) -> Image:
        #Create kernel
        self._create_kernel(kernel_size)

        self._adapt_array(image.image_array, image.shape, kernel_size)

        result = 0
        filtered_array = np.zeros(image.shape)

        if len(image.shape) == 2:        
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    for k in range(0, kernel_size):
                        for l in range(0, kernel_size):
                            result = self.adapted_array[i+k,j+l] * self.kernel[k,l] + result
                    
                    result = math.ceil((result/(kernel_size**2)))
                    filtered_array[i,j] = result
                    result = 0
        elif len(image.shape) == 3:
            for h in range(0, image.shape[2]):
                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        for k in range(0, kernel_size):
                            for l in range(0, kernel_size):
                                result = self.adapted_array[i+k,j+l,h] * self.kernel[k,l] + result
                        
                        result = math.ceil((result/(kernel_size**2)))
                        filtered_array[i,j,h] = result
                        result = 0

        filtered_array = np.uint8(filtered_array)

        return Image(None, filtered_array)

class MedianFilter(FilterTemplate):
    'Meadian Filter for Image processing'

    def __init__(self) -> None:
        pass

    def _create_kernel(self, size) -> None:
        self.kernel = np.ones((size,size))

    def _adapt_array(self, image_array: np.ndarray, shape: tuple ,kernel_size: int) -> None:
        center = math.floor(kernel_size/2)

        if len(shape) == 2:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2))
            self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center)] = image_array
        elif len(shape) == 3:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2, shape[2]))

            for i in range(0, shape[2]):
                self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center), i] = image_array[:,:,i]

    
    def operate_filter(self, image: Image, kernel_size: int = 2) -> Image:
        #Create kernel
        self._create_kernel(kernel_size)

        self._adapt_array(image.image_array, image.shape, kernel_size)

        result = np.zeros(kernel_size**2)
        filtered_array = np.zeros(image.shape)

        if len(image.shape) == 2:        
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    for k in range(0, kernel_size):
                        for l in range(0, kernel_size):
                            result[(kernel_size*k+l)] = (self.adapted_array[i+k,j+l])
                    
                    result = np.sort(result)
                    filtered_array[i,j] = result[int(np.floor((kernel_size**2)/2))]
        elif len(image.shape) == 3:
            for h in range(0, image.shape[2]):
                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        for k in range(0, kernel_size):
                            for l in range(0, kernel_size):
                                result[(kernel_size*k+l)] = (self.adapted_array[i+k,j+l,h])
                        
                        result = np.sort(result)
                        filtered_array[i,j,h] = result[int(np.floor((kernel_size**2)/2))]

        filtered_array = np.uint8(filtered_array)

        return Image(None, filtered_array)

class GaussianFilter(FilterTemplate):
    'Gaussian Filter for Image processing'

    def __init__(self, sigma: float = 0.5) -> None:
        self.sigma = sigma

    def _create_kernel(self, size: int) -> None:
        # Creates a 2D normalized, size x size, Gaussian Kernel
        self.sigma = 0.5
        x, y = np.mgrid[0:size, 0:size]
        #print("x original",x)
        #print("y original",y,"\n\n")
        x = x-size/2
        y = y-size/2
        #print("x modificado",x)
        #print("y modifica",y)
        kernel = np.exp( -( x**2 + y**2 ) / (2*self.sigma**2) ) * (1/(2*np.pi*self.sigma**2))

        self.kernel = kernel/kernel.sum()

    def _adapt_array(self, image_array: np.ndarray, shape: tuple ,kernel_size: int) -> None:
        center = math.floor(kernel_size/2)

        if len(shape) == 2:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2))
            self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center)] = image_array
        elif len(shape) == 3:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2, shape[2]))

            for i in range(0, shape[2]):
                self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center), i] = image_array[:,:,i]

    
    def operate_filter(self, image: Image, kernel_size: int = 2) -> Image:
        #Create kernel
        self._create_kernel(kernel_size)

        self._adapt_array(image.image_array, image.shape, kernel_size)

        result = 0
        filtered_array = np.zeros(image.shape)

        if len(image.shape) == 2:        
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    for k in range(0, kernel_size):
                        for l in range(0, kernel_size):
                            result = self.adapted_array[i+k,j+l] * self.kernel[k,l] + result
                    
                    result = math.ceil(result)
                    filtered_array[i,j] = result
                    result = 0
        elif len(image.shape) == 3:
            for h in range(0, image.shape[2]):
                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        for k in range(0, kernel_size):
                            for l in range(0, kernel_size):
                                result = self.adapted_array[i+k,j+l,h] * self.kernel[k,l] + result
                        
                        result = math.ceil(result)
                        filtered_array[i,j,h] = result
                        result = 0

        filtered_array = np.uint8(filtered_array)

        return Image(None, filtered_array)

class LaplacianFilter(FilterTemplate):
    'Laplacian Filter for Image processing'

    def __init__(self, positive_kernel: bool = True, kernel_type: int = 1) -> None:
        self.positive_kernel = positive_kernel
        self.kernel_type = kernel_type

    def _create_kernel(self, size) -> None:
        if size % 2 == 0:
            size += 1

        if self.positive_kernel:
            self.kernel = np.ones((size,size))
            aux = -1
        else:
            self.kernel = np.ones((size,size)) * -1
            aux = 1
        
        if self.kernel_type == 1:
            self.kernel[0,0], self.kernel[-1,-1], self.kernel[-1,0], self.kernel[0,-1] = 0,0,0,0
        print("Kernel Original",self.kernel,sep='\n',end='\n\n')

        self.kernel[size//2,size//2] = -1 * (np.sum(self.kernel) + aux)
        print("Kernel Modificado",self.kernel,sep='\n')
        #sigma = 0.4
        #x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
        #kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        #self.kernel = kernel / np.sum(np.abs(kernel))

    def _adapt_array(self, image_array: np.ndarray, shape: tuple ,kernel_size: int) -> None:
        center = math.floor(kernel_size/2)

        if len(shape) == 2:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2))
            self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center)] = image_array
        elif len(shape) == 3:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2, shape[2]))

            for i in range(0, shape[2]):
                self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center), i] = image_array[:,:,i]

    
    def operate_filter(self, image: Image, kernel_size: int = 2) -> Image:
        #Create kernel
        self._create_kernel(kernel_size)

        self._adapt_array(image.image_array, image.shape, kernel_size)

        result = 0
        filtered_array = np.zeros(image.shape)

        if len(image.shape) == 2:        
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    for k in range(0, kernel_size):
                        for l in range(0, kernel_size):
                            result = self.adapted_array[i+k,j+l] * self.kernel[k,l] + result
                    
                    result = math.ceil(result/(kernel_size**2))
                    filtered_array[i,j] = result
                    result = 0
        elif len(image.shape) == 3:
            for h in range(0, image.shape[2]):
                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        for k in range(0, kernel_size):
                            for l in range(0, kernel_size):
                                result = self.adapted_array[i+k,j+l,h] * self.kernel[k,l] + result
                        
                        result = math.ceil(result/(kernel_size**2))
                        filtered_array[i,j,h] = result
                        result = 0

        filtered_array = np.uint8(filtered_array)

        return Image(None, filtered_array)

class PrewitFilter(FilterTemplate):
    'Prewit Filter for Image processing'

    def __init__(self) -> None:
        pass

    def _create_kernel(self, size) -> None:
        self.kernel_horizontal = np.zeros((size,size))
        self.kernel_horizontal[:,0] = -1
        self.kernel_horizontal[:,size-1] = 1

        self.kernel_vertical = np.zeros((size,size))
        self.kernel_vertical[0,:] = -1
        self.kernel_vertical[size-1,:] = 1

    def _adapt_array(self, image_array: np.ndarray, shape: tuple ,kernel_size: int) -> None:
        center = math.floor(kernel_size/2)

        if len(shape) == 2:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2))
            self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center)] = image_array
        elif len(shape) == 3:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2, shape[2]))

            for i in range(0, shape[2]):
                self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center), i] = image_array[:,:,i]

    
    def operate_filter(self, image: Image, kernel_size: int = 2) -> Image:
        #Create kernel
        self._create_kernel(kernel_size)

        self._adapt_array(image.image_array, image.shape, kernel_size)

        result_v = 0
        result_h = 0
        filtered_array = np.zeros(image.shape)

        if len(image.shape) == 2:        
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    for k in range(0, kernel_size):
                        for l in range(0, kernel_size):
                            result_v = self.adapted_array[i+k,j+l] * self.kernel_vertical[k,l] + result_v
                            result_h = self.adapted_array[i+k,j+l] * self.kernel_horizontal[k,l] + result_h
                    
                    result_v = math.ceil((result_v/(kernel_size**2)))
                    result_h = math.ceil((result_h/(kernel_size**2)))
                    filtered_array[i,j] = np.sqrt(result_v**2 + result_h**2)
                    result_v, result_v = 0,0
        elif len(image.shape) == 3:
            for h in range(0, image.shape[2]):
                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        for k in range(0, kernel_size):
                            for l in range(0, kernel_size):
                                result_v = self.adapted_array[i+k,j+l,h] * self.kernel_vertical[k,l] + result_v
                                result_h = self.adapted_array[i+k,j+l,h] * self.kernel_horizontal[k,l] + result_h
                        
                        result_v = math.ceil((result_v/(kernel_size**2)))
                        result_h = math.ceil((result_h/(kernel_size**2)))
                        filtered_array[i,j,h] = np.sqrt(result_v**2 + result_h**2)
                        result_v, result_v = 0,0

        filtered_array = np.uint8(filtered_array)

        return Image(None, filtered_array)

class SobelFilter(FilterTemplate):
    'Sobel Filter for Image processing'

    def __init__(self) -> None:
        pass

    def _create_kernel(self, size) -> None:
        self.kernel_horizontal = np.zeros((size,size))
        self.kernel_horizontal[:,0] = -1
        self.kernel_horizontal[size//2,0] = -2
        self.kernel_horizontal[:,size-1] = 1
        self.kernel_horizontal[size//2,size-1] = 2

        self.kernel_vertical = np.zeros((size,size))
        self.kernel_vertical[0,:] = -1
        self.kernel_vertical[0,size//2] = -2
        self.kernel_vertical[size-1,:] = 1
        self.kernel_vertical[size-1,size//2] = 2

    def _adapt_array(self, image_array: np.ndarray, shape: tuple ,kernel_size: int) -> None:
        center = math.floor(kernel_size/2)

        if len(shape) == 2:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2))
            self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center)] = image_array
        elif len(shape) == 3:
            self.adapted_array = np.zeros((shape[0] + center * 2, shape[1] + center * 2, shape[2]))

            for i in range(0, shape[2]):
                self.adapted_array[(0+center):(shape[0]+center), (0+center):(shape[1]+center), i] = image_array[:,:,i]

    
    def operate_filter(self, image: Image, kernel_size: int = 2) -> Image:
        #Create kernel
        self._create_kernel(kernel_size)

        self._adapt_array(image.image_array, image.shape, kernel_size)

        result_v = 0
        result_h = 0
        filtered_array = np.zeros(image.shape)

        if len(image.shape) == 2:        
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    for k in range(0, kernel_size):
                        for l in range(0, kernel_size):
                            result_v = self.adapted_array[i+k,j+l] * self.kernel_vertical[k,l] + result_v
                            result_h = self.adapted_array[i+k,j+l] * self.kernel_horizontal[k,l] + result_h
                    
                    result_v = math.ceil((result_v/(kernel_size**2)))
                    result_h = math.ceil((result_h/(kernel_size**2)))
                    filtered_array[i,j] = np.sqrt(result_v**2 + result_h**2)
                    result_v, result_v = 0,0
        elif len(image.shape) == 3:
            for h in range(0, image.shape[2]):
                for i in range(0, image.shape[0]):
                    for j in range(0, image.shape[1]):
                        for k in range(0, kernel_size):
                            for l in range(0, kernel_size):
                                result_v = self.adapted_array[i+k,j+l,h] * self.kernel_vertical[k,l] + result_v
                                result_h = self.adapted_array[i+k,j+l,h] * self.kernel_horizontal[k,l] + result_h
                        
                        result_v = math.ceil((result_v/(kernel_size**2)))
                        result_h = math.ceil((result_h/(kernel_size**2)))
                        filtered_array[i,j,h] = np.sqrt(result_v**2 + result_h**2)
                        result_v, result_v = 0,0

        filtered_array = np.uint8(filtered_array)

        return Image(None, filtered_array)







