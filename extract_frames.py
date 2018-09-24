import os
import sys
import cv2

# Try to create a folder if does not exist. Ignore if it does.
try:
    os.mkdir('./Frames')
except:
    pass

# Function to extract images from a particular input video. Here count_tag
# represents the tag number in continutation from previos video to maintain
# numeric order.
def extractImages(pathIn, pathOut, count_tag):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    prev_frame_count = 0
    while success:

        # Store the frames to the output path
        cv2.imwrite(pathOut + str(count_tag + count) + ".jpg", image)     
        
        # Skip the video by 10 seconds 
        # (can reduce time if more training data required)
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*10000))    
        
        # Read the frame at current position
        success,image = vidcap.read()        
        
        count = count + 1

        # To handle the problem in OpenCV where it is not
        # able to read last frame and throw read error
        # So checking if frame count does not change between
        # two intervals, then video has ended.
        frame_count = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
        if(prev_frame_count == frame_count):
            break
        else:
            prev_frame_count = frame_count
        
    # return next frame's start count
    return (count+count_tag)

def main():

    # Read the input category
    if len(sys.argv) > 1:
        class_type = sys.argv[1]
    else:
        print("Usage: python", sys.argv[0], "[folder category]")
        print("folder category: indoor or outdoor")
        return

    count = 1

    # Scan the current directory for webm and mp4 video files
    for entry in os.scandir('./'): 
        if(entry.path.endswith('webm') or entry.path.endswith('mp4')):
            print("Processing file", entry.path, "from count",count)
            count = extractImages(entry.path, 'Frames/' + class_type, count)


if __name__ == "__main__":
    main()