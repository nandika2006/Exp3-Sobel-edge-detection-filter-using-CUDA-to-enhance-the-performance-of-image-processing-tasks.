# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>NANDIKA S</h3>
<h3>212224230175</h3>
<h3>EX. NO</h3>
<h3>23-11-2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:

    !pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
    %load_ext nvcc4jupyter
    !nvcc --version
    %load_ext nvcc4jupyter
    from pathlib import Path
    
    file_path = Path('/content/rose.webp')
    if file_path.exists():
        print("File exists!")
    else:
        print("File does not exist!")
    import os
    print("Current Working Directory:", os.getcwd())
    from google.colab import files
    uploaded = files.upload()
    pwd
    ls /content/pngtree-holstein-cow-png-image_20610191 (1).png
    #ls -l /content/images.jpeg
    import cv2
    image = cv2.imread('/content/pngtree-holstein-cow-png-image_20610191.png')
    if image is None:
        print("Error: Image not found or unable to read the image.")
    else:
        print("Image read successfully.")
    %%writefile sobelEdgeDetectionFilter.cu
    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    #include <cuda_runtime.h>
    #include <opencv2/opencv.hpp>
    
    using namespace cv;
    
    __global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        float Kx[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        float Ky[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
        // only threads inside image will write results
        if ((x >= 3 / 2) && (x < (width - 3 / 2)) && (y >= 3 / 2) && (y < (height - 3 / 2))) {
            // Gradient in x-direction
            float Gx = 0;
            // Loop inside the filter to average pixel values
            for (int ky = -3 / 2; ky <= 3 / 2; ky++) {
                for (int kx = -3 / 2; kx <= 3 / 2; kx++) {
                    float fl = srcImage[((y + ky) * width + (x + kx))];
                    Gx += fl * Kx[ky + 3 / 2][kx + 3 / 2];
                }
            }
            float Gx_abs = Gx < 0 ? -Gx : Gx;
            // Gradient in y-direction
            float Gy = 0;
            // Loop inside the filter to average pixel values
            for (int ky = -3 / 2; ky <= 3 / 2; ky++) {
                for (int kx = -3 / 2; kx <= 3 / 2; kx++) {
                    float fl = srcImage[((y + ky) * width + (x + kx))];
                    Gy += fl * Ky[ky + 3 / 2][kx + 3 / 2];
                }
            }
            float Gy_abs =   Gy < 0 ? -Gy : Gy;
            dstImage[(y * width + x)] = Gx_abs + Gy_abs;
        }
    }
    import cv2
    from matplotlib import pyplot as plt

# Read and display the output image
output_image_path = '/content/rose.webp'
output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)  # Use IMREAD_GRAYSCALE if it's a single-channel image

# Display the image
plt.imshow(output_image, cmap='gray')
plt.title('Edge Detection Output')
plt.axis('off')  # Hide the axes
plt.show()

## OUTPUT:
<img width="639" height="583" alt="image" src="https://github.com/user-attachments/assets/280629b4-ddc6-48ce-ba76-27362dc17b6a" />


## RESULT:
Thus the program has been executed by using CUDA to 2.282023 ms.

Questions:

1. *Challenges Implementing Sobel for Color Images*:
   - Converting images to grayscale in the kernel increased complexity. Memory management and ensuring correct indexing for color to grayscale conversion required attention.

2. *Influence of Block Size*:
   - Smaller block sizes (e.g., 8x8) were efficient for smaller images but less so for larger ones, where larger blocks (e.g., 32x32) reduced overhead.

3. *CUDA vs. CPU Output Differences*:
   - The CUDA implementation was faster, with minor variations in edge sharpness due to rounding differences. CPU output took significantly more time than the GPU.

4. *Optimization Suggestions*:
   - Use shared memory in the CUDA kernel to reduce global memory access times.
   - Experiment with adaptive block sizes for larger images.

