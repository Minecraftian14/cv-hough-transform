from data_generator import *
from hough_lines import *


#define my_roll_number and implement count_hough_lines is hough_lines.py

def test():
    roll = my_roll_number()
    name = "Anirudh Sharma"

    img1 = blank_ruled_paper(roll)
    img2 = printed_paper(roll, name, roll)
    img3 = handwritten_paper(roll, name, roll)
     
    #Code to visualize the generated image.
    cv2.imshow("image", img1)
    cv2.waitKey(0)
    cv2.imshow("image", img2)
    cv2.waitKey(0)
    cv2.imshow("image", img3)
    cv2.waitKey(0)

    print("Blank ruled lines:", count_hough_lines(img1))
    print("Printed lines:", count_hough_lines(img2))
    print("Handwritten lines:", count_hough_lines(img3))

if __name__ == "__main__":
    test()