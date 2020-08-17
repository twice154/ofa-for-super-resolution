import cv2

# Opens the Video file
cap = cv2.VideoCapture('./basketball.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps: "+str(fps))
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("num_frames: "+str(num_frames))
# exit()
# cap.set(cv2.CAP_PROP_FPS, 1)
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    # basketball
        # 포맷: H.264 HD (1-1-1)
        # 해상도: 1920 x 1080, 16:9
        # 인코딩된 FPS: 25
        # 데이터 크기: 4.81GB
        # 데이터율: 1.84Mbit/초
    # bolt_100m
        # 포맷: H.264 HD (1-1-1)
        # 해상도: 1920 x 1080, 16:9
        # 인코딩된 FPS: 25
        # 데이터 크기: 364.9MB
        # 데이터율: 1.82Mbit/초
    # bolt_200m
        # 포맷: H.264 HD (1-1-1)
        # 해상도: 1920 x 1080, 16:9
        # 인코딩된 FPS: 25
        # 데이터 크기: 248.8MB
        # 데이터율: 1.84Mbit/초
    # jtbc_newsroom
        # 포맷: H.264 HD (1-1-1), 스테레오(L R), AAC, 44100 Hz
        # 해상도: 1280 x 720, 16:9
        # 인코딩된 FPS: 29.97
        # 데이터 크기: 194.3MB
        # 데이터율: 730.07kbit/s
    # starcraft
        # 포맷: H.264 HD (1-1-1)
        # 해상도: 1920 x 1080, 16:9
        # 인코딩된 FPS: 29.97
        # 데이터 크기: 359MB
        # 데이터율: 1.44Mbit/초
    if i % 1 == 0:  # 몇 개 프레임마다 저장할 것인가...
        cv2.imwrite('./basketball/'+'basketball_'+str(i)+'.png',frame)
        print("frame #"+str(i)+" is extracted and saved.")
    i+=1

cap.release()
cv2.destroyAllWindows()
