import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    print("Hello World!")
    pic = Image.open("data\\jj_and_omer.jpeg")
    print(pic)
    print(type(pic))
    pic_arr = np.asarray(pic)
    plt.imshow(pic_arr)
    pic_red = pic_arr[:,:,0]
    print(pic_red)

if __name__ == "__main__":
    main()