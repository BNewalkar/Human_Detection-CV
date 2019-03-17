# importing required libraries
import numpy as np
from skimage.io import imread
from PIL import Image
import math
import sys

class HOG:
    def __init__(self, image_name):
        self.image_name = image_name

    # function to convert color image to greyscale
    def greyscale_operation(self, ip_img):
        op_img = np.zeros((ip_img.shape[0], ip_img.shape[1]))
        for i in range(ip_img.shape[0]):
            for j in range(ip_img.shape[1]):
                op_img[i][j] = np.round_(0.299*ip_img[i][j][0] + 0.587*ip_img[i][j][1] + 0.114*ip_img[i][j][2])
        return op_img

    # function for implementing gradient operation using prewitt's edge detector
    def gradient_operation(self, ip_img):
        # prewitt's vertical and horizontal kernels
        Gx_kernel = ([[-1, 0, 1]] * 3)
        Gy_kernel = ([1, 1, 1], [0, 0, 0], [-1, -1, -1])

        Gx = np.zeros((ip_img.shape[0], ip_img.shape[1]))
        Gy = np.zeros((ip_img.shape[0], ip_img.shape[1]))
        G = np.zeros((ip_img.shape[0], ip_img.shape[1]))
        Theta = np.zeros((ip_img.shape[0], ip_img.shape[1]))

        # calculating Gx and Gy using prewitt's edge detector
        for i in range(ip_img.shape[0]):
            for j in range(ip_img.shape[1]):
                # pixels for which part of the prewitt's mask goes outside of the image border
                if i < 1 or j < 1 or ip_img.shape[0] - i <= 1 or ip_img.shape[1] - j <= 1:
                    continue
                else:
                    arr1 = ip_img[i - 1:i + 2, j - 1:j + 2]
                    Gx[i][j] = np.sum(np.multiply(arr1, Gx_kernel))/3
                    Gy[i][j] = np.sum(np.multiply(arr1, Gy_kernel))/3
                    G[i][j] = round(math.sqrt(Gx[i][j]**2 + Gy[i][j]**2)/math.sqrt(2))
                    if Gx[i][j] == 0 and Gy[i][j] == 0:
                        Theta[i][j] = 0
                    else:
                        theta = np.arctan2(Gy[i][j], Gx[i][j])*180/np.pi
                        if theta < 0:
                            theta += 180
                        if theta >= 170:
                            theta -= 180
                        Theta[i][j] = theta

        return G, Theta

    # HoG implementation
    def hog_operation(self, ip_img, ip_theta):
        # HoG bins table
        bins_table = {0:[-10,10], 20:[10,30], 40:[30,50], 60:[50,70], 80:[70,90], 100:[90,110], 120:[110,130], 140:[130, 150], 160:[150,170]}
        histo_center = np.zeros((ip_img.shape[0], ip_img.shape[1]))

        # First the bin centers for each pixel are identified
        for i in range(ip_theta.shape[0]):
            for j in range(ip_theta.shape[1]):
                for x in bins_table.keys():
                    if bins_table[x][0] <= ip_theta[i][j] < bins_table[x][1]:
                        histo_center[i][j] = x

        # hog histograms for each cell are created
        histo_list = np.zeros((int(ip_img.shape[0]/8), int(ip_img.shape[1]/8), 9))

        a = 0
        k = 0
        for x in range(int(ip_theta.shape[0]/8)):
            b = 0
            c = 0
            for y in range(int(ip_theta.shape[1]/8)):
                for i in range(a, a+8):
                    for j in range(b, b+8):
                        bin_no = int(histo_center[i][j]/20)
                        if ip_theta[i][j] == histo_center[i][j]:
                            histo_list[k][c][bin_no] += ip_img[i][j]
                        else:
                            diff1 = abs(bins_table[histo_center[i][j]][0] - ip_theta[i][j])
                            diff2 = abs(bins_table[histo_center[i][j]][1] - ip_theta[i][j])
                            if ip_theta[i][j] < histo_center[i][j]:
                                histo_list[k][c][bin_no] += (diff2/20)*ip_img[i][j]
                                histo_list[k][c][bin_no - 1] += (diff1/20)*ip_img[i][j]
                            else:
                                histo_list[k][c][bin_no] += (diff1/20)*ip_img[i][j]
                                if bin_no == 8:
                                    histo_list[k][c][0] += (diff2/20)*ip_img[i][j]
                                else:
                                    histo_list[k][c][bin_no + 1] += (diff2/20)*ip_img[i][j]
                b += 8
                c += 1
            a += 8
            k += 1

        # block_array will store the normalized histogram block wise
        block_array = np.zeros((histo_list.shape[0] - 1, histo_list.shape[1] - 1, 36))

        # hog_discriptor will create a one dimensional array of all the normalized hirtograms
        hog_discriptor_op = []
        for i in range(histo_list.shape[0] - 1):
            for j in range(histo_list.shape[1] - 1):
                l1 = histo_list[i][j]
                l2 = histo_list[i+1][j]
                l3 = histo_list[i][j+1]
                l4 = histo_list[i+1][j+1]

                norm_factor = math.sqrt(np.sum(np.square(l1)) +  np.sum(np.square(l2)) + np.sum(np.square(l3)) + np.sum(np.square(l4)))
                np.seterr(divide='ignore', invalid='ignore')
                l1_, l2_, l3_, l4_ = np.true_divide(l1, norm_factor), np.true_divide(l2, norm_factor), np.true_divide(l3, norm_factor), np.true_divide(l4, norm_factor)
                block_array[i][j] = np.concatenate((l1_, l2_, l3_, l4_))
                hog_discriptor_op.extend(l1_.tolist())
                hog_discriptor_op.extend(l2_.tolist())
                hog_discriptor_op.extend(l3_.tolist())
                hog_discriptor_op.extend(l4_.tolist())

        return np.array(hog_discriptor_op).reshape((1, len(hog_discriptor_op)))


    # main function
    def hog(self):
        image_name = self.image_name
        img = imread(image_name)
        gs_img = self.greyscale_operation(img)
        grad_img, theta = self.gradient_operation(gs_img)
        hog_discriptor = self.hog_operation(grad_img, theta)
        return hog_discriptor
