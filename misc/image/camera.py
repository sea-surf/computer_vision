import cv2 as cv

# Open video
video_path = ".../films/movie.mkv"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while True:
    ret, frame = cap.read()

    # If the video ends, ret will be False
    if not ret:
        print("End of video file. Exiting ...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imshow("MKV Playback", gray)

    if cv.waitKey(25) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()