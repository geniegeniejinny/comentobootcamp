import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_red(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print("이미지를 찾을 수 없습니다:", img_path)
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    result = cv2.bitwise_and(image, image, mask=mask)

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Red Filtered")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    detect_red("sample.jpg")
