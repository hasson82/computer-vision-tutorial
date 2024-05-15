import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    tens = np.ones((5,5))*10
    print(tens)

    np.random.seed(101)
    arr = np.random.randint(0,100,(5,5))
    print(arr)
    print(arr.max())
    print(arr.min())
    pic = Image.open("data\\jj_and_omer.jpeg")
    pic_arr = np.asarray(pic)
    pic_copy = pic_arr.copy()
    pic_copy[:,:,1] = 0
    pic_copy[:,:,2] = 0
    print(pic_copy.shape)
    plt.imshow(pic_copy)
    plt.show()
    plt.imshow(pic_arr)
    plt.show()


if __name__ == "__main__":
    main()