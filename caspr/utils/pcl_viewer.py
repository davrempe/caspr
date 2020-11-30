'''
Mouse Controls:
- [Scroll]zoom in/out
- [Left click & Drag] Rotate view
- [Right click & Drag] Pan view
Key Controls:
- [ESC] Exit
- [s] Save screenshot
- [a] Show all frames at once
- [p] Pause/play animation
- [t] Toggle visible sequences
- [<-][->] Step back/forward one frame
- [-][+] Decrease/increase point size
'''

import sys, os, argparse, cv2
from time import sleep

from tk3dv.common import drawing, utilities
import tk3dv.nocstools.datastructures as ds
from tk3dv.nocstools import obj_loader
from PyQt5.QtWidgets import QApplication
import PyQt5.QtCore as QtCore
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
import numpy as np

from tk3dv.pyEasel import *
from EaselModule import EaselModule
from Easel import Easel
import OpenGL.GL as gl

class CloseEasel(Easel):
     def closeEvent(self, event):
        self.stop()
        for Mod in self.Modules:
            Mod.__del__()

class PCLViewer(EaselModule):
    def __init__(self, pcl_seq, 
                rgb_seq=None, 
                cameras=None,
                fps=60,
                autoplay=True,
                draw_cubes=True,
                out_path=None):
        '''
        - N point cloud sequences of the same length of steps (can have different number of points) : list of lists [[np.array(N x 3)]]
        - N rgb color sequences (optional) : list of lists [[np.array(N x 3)]]
        - list of N camera extrinsics : [Nx4x4 [R|t]] numpy array that represents the transformation cam2world
        '''
        super().__init__()
        # sequence of raw depth image data
        self.pcl_seq = pcl_seq
        self.rgb_seq = rgb_seq
        self.draw_cubes = draw_cubes
        # point set objects to visualize
        self.pt_set_seq = []

        self.PointSize = 3
        self.seq_len = len(pcl_seq[0])
        for seq_idx in range(len(self.pcl_seq)):
            if len(self.pcl_seq[seq_idx]) != self.seq_len:
                print('All sequences must be the same length to visualize!')
                exit()
        self.cur_frame = 0

        self.num_seq = len(self.pcl_seq)
        self.num_frames_per_seq = len(self.pcl_seq[0])
        self.cur_seq = self.num_seq
        self.num_display = self.num_seq + 2
        print('Number of sequences: ' + str(self.num_seq))

        self.fps = fps
        self.autoplay = autoplay
        self.draw_all = False
        self.take_ss = False # whether we're currently taking a screenshot
        self.ss_ctr = 0
        self.out_path = out_path
        if self.out_path is None:
            self.out_path = '.'

        self.need_update_step = autoplay

        self.cam_colors = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float)

        self.cameras = None
        if cameras is not None:
            self.cameras = []
            for extrins_list in cameras:
                cur_extrins = []
                for T in extrins_list:
                    extrins = ds.CameraExtrinsics(rotation=T[:3,:3].T, translation=T[:3,3])
                    # print(extrins)
                    cur_extrins.append(ds.Camera(Extrinsics=extrins))
                self.cameras.append(cur_extrins)

        print('Viz sequence of length: ' + str(self.seq_len))


    def init(self, argv=None):
        # create point sets for each frame of each sequence
        for seq_idx in range(self.num_seq):
            cur_pcl_seq = self.pcl_seq[seq_idx]
            self.pt_set_seq.append([])
            for frame_idx in range(self.seq_len):
                cur_pcl = cur_pcl_seq[frame_idx]
                cur_rgb = None
                if self.rgb_seq is not None and self.rgb_seq[seq_idx] is not None:
                    cur_rgb = self.rgb_seq[seq_idx][frame_idx]
            
                cur_pts = ds.PointSet3D()
                # print(cur_pcl.shape)
                if cur_pcl.shape[0] > 0:
                    cur_pts.Points = cur_pcl.astype(np.float)
                    if cur_rgb is not None:
                        cur_pts.Colors = cur_rgb.astype(np.float)
                    else:
                        cur_pts.Colors = np.zeros_like(cur_pts.Points, dtype=np.float)
                cur_pts.update()
                self.pt_set_seq[-1].append(cur_pts)


    def step(self):
        if not self.need_update_step:
            return
        else:
            self.need_update_step = self.autoplay

        startTime = utilities.getCurrentEpochTime()

        self.cur_frame = (self.cur_frame + 1) % self.seq_len

        endTime = utilities.getCurrentEpochTime()
        ElapsedTime = (endTime - startTime)

        if self.autoplay:
            desired_step_time = (1.0 / self.fps)*1e6
            if ElapsedTime < desired_step_time:
                sleep((desired_step_time - ElapsedTime) / 1e6)

    def step_back(self):
        if not self.need_update_step:
            return
        else:
            self.need_update_step = self.autoplay
        self.cur_frame = (self.cur_frame - 1) % self.seq_len

    def drawLine(self, start_pt, end_pt, LineWidth=5.0, Color=None):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        gl.glPushAttrib(gl.GL_LINE_BIT)
        gl.glLineWidth(LineWidth)
        gl.glBegin(gl.GL_LINES)
        if Color is None:
            gl.glColor3f(1.0, 0.0, 0.0)
        else:
            gl.glColor3fv(Color)
        gl.glVertex3f(start_pt[0], start_pt[1], start_pt[2])
        gl.glVertex3f(end_pt[0], end_pt[1], end_pt[2])

        gl.glEnd()

        gl.glPopAttrib()
        gl.glPopMatrix()

    def draw(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()

        ScaleFact = 1000
        gl.glScale(ScaleFact, ScaleFact, ScaleFact)

        if self.draw_cubes:
            drawing.drawUnitWireCube(2.0, False, WireColor=(0.0, 1.0, 0.0))

            gl.glPushMatrix()
            gl.glTranslate(1.0, 0.0, 0.0)
            drawing.drawUnitWireCube(2.0, False, WireColor=(1.0, 0.0, 0.0))
            gl.glPopMatrix()

        if self.draw_all:
            for frame_idx in range(self.num_frames_per_seq):
                for seq_idx in range(self.num_seq):
                        self.pt_set_seq[seq_idx][frame_idx].draw(self.PointSize)
        else:
            if self.cur_seq == self.num_seq:
                for seq_idx in range(self.num_seq):
                    self.pt_set_seq[seq_idx][self.cur_frame].draw(self.PointSize)
            elif self.cur_seq < self.num_seq:
                self.pt_set_seq[self.cur_seq][self.cur_frame].draw(self.PointSize)

        if self.cameras is not None:
            draw_frames = [self.cur_frame]
            if self.draw_all:
                draw_frames = range(self.num_frames_per_seq)
            for cam_frame_idx in draw_frames:
                for cam_idx in range(len(self.cameras)):
                    self.cameras[cam_idx][cam_frame_idx].draw(Color=self.cam_colors[cam_idx % len(self.cam_colors)], 
                                                              CubeSide=-0.2, isDrawDir=True, Length=0.3)

            # draw cam traj
            for cam_idx in range(len(self.cameras)):
                for cam_frame_idx in range(len(self.cameras[cam_idx])-1):
                    self.drawLine(self.cameras[cam_idx][cam_frame_idx].Extrinsics.Translation, self.cameras[cam_idx][cam_frame_idx+1].Extrinsics.Translation, \
                            Color=self.cam_colors[cam_idx % len(self.cam_colors)], LineWidth=1.0)

        gl.glPopMatrix()

        if self.take_ss:
            x, y, width, height = gl.glGetIntegerv(gl.GL_VIEWPORT)
            # print("Screenshot viewport:", x, y, width, height)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)

            data = gl.glReadPixels(x, y, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            SS = np.frombuffer(data, dtype=np.uint8)
            SS = np.reshape(SS, (height, width, 4))
            SS = cv2.flip(SS, 0)
            SS = cv2.cvtColor(SS, cv2.COLOR_BGRA2RGBA)
            ss_out_path = os.path.join(self.out_path,'screenshot_' + str(self.ss_ctr).zfill(6) + '.png' )
            cv2.imwrite(ss_out_path, SS)
            self.ss_ctr = self.ss_ctr + 1
            self.take_ss = False

            print('[ INFO ]: Done saving.')
            sys.stdout.flush()

    def keyPressEvent(self, a0: QKeyEvent):
        if a0.key() == QtCore.Qt.Key_Plus:  # Increase or decrease point size
            if self.PointSize < 20:
                self.PointSize = self.PointSize + 1

        if a0.key() == QtCore.Qt.Key_Minus:  # Increase or decrease point size
            if self.PointSize > 1:
                self.PointSize = self.PointSize - 1

        if a0.key() == QtCore.Qt.Key_Left:  # Go back a frame
            self.need_update_step = True
            self.step_back()
        if a0.key() == QtCore.Qt.Key_Right:  # Go forward frames
            self.need_update_step = True
            self.step()

        if a0.key() == QtCore.Qt.Key_T: # toggle which sequence is showing
            self.cur_seq += 1
            self.cur_seq %= self.num_display

        if a0.key() == QtCore.Qt.Key_P:
            self.autoplay = not self.autoplay
            self.need_update_step = self.autoplay

        if a0.key() == QtCore.Qt.Key_A: # show all frames at once
            self.draw_all = not self.draw_all

        if a0.key() == QtCore.Qt.Key_S:
            print('[ INFO ]: Taking snapshot. This might take a while...')
            sys.stdout.flush()
            self.take_ss = True


def viz_pcl_seq(pcl_seq, 
                rgb_seq=None, 
                fps=60,
                autoplay=True,
                cameras=None,
                draw_cubes=True,
                out_path=None):
    ''' 
    Visualize one or more sequences of point clouds with mapped colors (optional).

    If autoplay is true, will play at the given framerate, otherwise
    Can step through frames with keys.

    Inputs:
    - pcl_seq : List of point cloud sequences, list of lists of np.arrays of size Nx3 where N may be differenct across frames and sequences.
    - rbg_seq : Same as pcl_seq, but with point colors instead of locations
    - fps     : Frame rate.
    - autoplay: If true, will automatically start playing. Otherwise frame stepping is controlled with arrow keys <- ->
    - cameras : List of camera transform sequences to visualize, list of np.arrays(F x 4 x 4) where F is the number of frames of each sequence
    - draw_cubes : if true draws the NOCS cube
    - out_path : path to save screenshots to (by default saves to '.')
    '''
    app = QApplication([''])
    view = PCLViewer(pcl_seq, rgb_seq=rgb_seq, cameras=cameras, fps=fps, 
                              autoplay=autoplay, draw_cubes=draw_cubes, 
                              out_path=out_path)
    mainWindow = CloseEasel([view])
    mainWindow.isUpdateEveryStep = autoplay
    mainWindow.show()
    app.exec_()