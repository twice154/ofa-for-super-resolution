import cv2

# Opens the Video file
cap = cv2.VideoCapture('./starcraft.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: "+str(fps))
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("num_frames: "+str(num_frames))
exit()
# cap.set(cv2.CAP_PROP_FPS, 1)
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i % 30 == 0:  # 몇 개 프레임마다 저장할 것인가. 일단은 1초에 1개씩 하는걸로...
        cv2.imwrite('./starcraft/'+'starcraft_'+str(i)+'.png',frame)
        print("frame #"+str(i)+" is extracted and saved.")
    i+=1

cap.release()
cv2.destroyAllWindows()
