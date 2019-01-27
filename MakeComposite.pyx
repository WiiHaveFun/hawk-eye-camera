import cv2

cpdef unsigned char[:, :, :] make_composite(unsigned char [:, :, :] blank_image, unsigned char [:, :, :] blank_image2, unsigned char [:, :, :] blank_image3, unsigned char [:, :, :] blank_image4, int height, int width):
	
	for y in range(height):
		for x in range(width):
			if(blank_image2[y][x][3] == 255):
				blank_image[y][x] = blank_image2[y][x]
	for y in range(height):
		for x in range(width):
			if(blank_image3[y][x][3] == 255):
				blank_image[y][x] = blank_image3[y][x]
	for y in range(height):
		for x in range(width):
			if(blank_image4[y][x][3] == 255):
				blank_image[y][x] = blank_image4[y][x]
	return blank_image
