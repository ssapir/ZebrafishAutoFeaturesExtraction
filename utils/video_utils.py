import cv2  # pip install opencv-contrib-python, doc: https://pypi.org/project/opencv-contrib-python/
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter  # important need ffmpeg.exe
import numpy as np
import os.path


def release(video, visualize_movie=False):
    video.release()
    if visualize_movie:
        cv2.destroyAllWindows()
    print("Windows closed (released)")


class VideoFromRaw:
    # FRAME_ROWS = 2336
    # FRAME_COLS = 1500
    # todo can read from txt?
    FRAME_ROWS = 896
    FRAME_COLS = 900
    FPS = 500
    RELEASE_FRAMES = 45000  # every this size, recreate memmap to release previous mem allocation

    def __init__(self, fullname, start_frame=None):  # start_frame starts from 1
        self.fullname = fullname
        self.video = None
        self.n_frames = 0
        self.create_file()
        print(start_frame, self.n_frames)
        if start_frame is not None and 1 <= start_frame <= self.n_frames:
            self.curr_frame_ind = start_frame - 1
        else:
            self.curr_frame_ind = 0

    def create_file(self):
        if hasattr(self, 'video') and self.video is not None:
            del self.video  # delete will release it
        try:
            # video is a map, in the form of numpy array
            self.video = np.memmap(self.fullname, dtype=np.uint8, mode='r').reshape([-1, self.FRAME_COLS, self.FRAME_ROWS])
            self.n_frames = self.video.shape[0]
            print("VideoFromRaw: created with shape ", self.video.shape)
        except Exception as e:
            self.video = None
            self.n_frames = 0
            print(e)

    def readFrame(self, curr_frame_ind):
        # release memory if needed before reading further
        if self.isOpened() and curr_frame_ind % self.RELEASE_FRAMES == 0:
            self.create_file()
        # frame number 0 till self.video.shape[2]-1
        if self.isOpened() and curr_frame_ind < self.n_frames:  # num of frames is 3rd
            ok = True
            frame = self.video[curr_frame_ind, :, :]
            return ok, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            print("Reached the end of ", self.video.shape)
            return False, np.array([])

    def read(self):
        # release memory if needed before reading further
        if self.isOpened() and self.curr_frame_ind % self.RELEASE_FRAMES == 0:
            self.create_file()
        # frame number 0 till self.video.shape[2]-1
        if self.isOpened() and self.curr_frame_ind < self.n_frames:  # num of frames is 3rd
            ok = True
            frame = self.video[self.curr_frame_ind, :, :]
            self.curr_frame_ind += 1
            return ok, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            print("Reached the end of ", self.video.shape)
            return False, np.array([])

    def isOpened(self):
        return self.video is not None

    def get_fps(self):
        return self.FPS

    def get_num_of_frames(self):
        return self.n_frames

    def get_curr_frame_num(self):
        return self.curr_frame_ind + 1  # start index from 1 not zero

    def release(self):
        del self.video  # release to allow GC


def open_raw(inputfolder, vidname, start_frame=None):
    fullname = os.path.join(inputfolder, vidname)
    video = VideoFromRaw(fullname, start_frame=start_frame)
    if not video.isOpened():
        print("Could not open video ", fullname)
        return video, -1, False, -1, -1

    fps = video.get_fps()
    n_frames = video.get_num_of_frames()
    n_curr_frame = video.get_curr_frame_num()

    return video, fps, True, n_frames, n_curr_frame


def open(inputfolder, vidname, start_frame=None):
    if vidname.lower().endswith(".avi"):
        return open_avi(inputfolder, vidname, start_frame=start_frame)
    elif vidname.lower().endswith(".raw"):
        return open_raw(inputfolder, vidname, start_frame=start_frame)
    else:
        print("Unsupported file type", vidname)
        return None, -1, False, -1, -1


def open_avi(inputfolder, vidname, start_frame=None):
    fullname = os.path.join(inputfolder, vidname)
    video = cv2.VideoCapture(fullname)

    if not video.isOpened():
        print("Could not open video ", fullname)
        return video, -1, False, -1, -1

    fps = video.get(cv2.CAP_PROP_FPS)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame is not None and 1 <= start_frame <= n_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)

    return video, fps, True, n_frames, int(video.get(cv2.CAP_PROP_POS_FRAMES)) + 1


def create_video_with_local_ffmpeg_exe(name, video_frames, fps):
    ny, nx = (video_frames[0].shape[1], video_frames[0].shape[0])
    prev_backend = plt.get_backend()
    plt.switch_backend("agg")
    dpi = 100  # from examples, not sure what it is
    fig = plt.figure(frameon=False, figsize=(nx / dpi, ny / dpi))
    ax = fig.add_subplot(111)

    writer = FFMpegWriter(fps=fps, codec="h264")
    with writer.saving(fig, name, dpi=dpi), np.errstate(invalid="ignore"):
        for image in video_frames:
            ax.imshow(image)  # todo colors!
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.axis("off")
            ax.invert_yaxis()
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            writer.grab_frame()
            ax.clear()
    plt.switch_backend(prev_backend)


def create_video_with_cv2_exe(name, video_frames, fps):
    ny, nx = (video_frames[0].shape[1], video_frames[0].shape[0])
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (ny, nx))
    for frame in video_frames:
        out.write(frame.astype('uint8'))
    out.release()
