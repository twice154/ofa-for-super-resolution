import cv2
from matplotlib import pyplot as plt
import os

file_list = os.listdir("./bolt_100m_25fps")
file_list = sorted(file_list,key=lambda x: int(os.path.splitext(x)[0]))

difference_list = []

for i in range(len(file_list)-1):
    previous_frame = file_list[i]
    current_frame = file_list[i+1]

    previous_image = cv2.imread("./bolt_100m_25fps/" + previous_frame)
    current_image = cv2.imread("./bolt_100m_25fps/" + current_frame)
    color = ('b', 'g', 'r')

    difference = 0

    for j, col in enumerate(color):
        previous_histr = cv2.calcHist([previous_image],[j],None,[256],[0,256])  # [256, 1]
        current_histr = cv2.calcHist([current_image],[j],None,[256],[0,256])  # [256, 1]
        difference_histr = previous_histr - current_histr
        difference_histr = difference_histr * difference_histr

        difference += sum(difference_histr)

    difference_list.append(difference)

    print("frame #" + str(i) + " is processed.")

plt.plot(difference_list)
plt.show()
