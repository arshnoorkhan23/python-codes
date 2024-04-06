from skimage import io
import os
from matplotlib import pyplot as plt
import numpy as np
# from pil import image
myimage_path=os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)),'images'),'1.jpg')
my_image=io.imread(myimage_path)
my_image[400:600,200:400,:]=[255,0,0]


# img1=image.open(myimage_path)
# img1.show()

# a=np.random.random([500,500])
plt.imshow(my_image)
plt.show()

# print(a)
# print('aa2')