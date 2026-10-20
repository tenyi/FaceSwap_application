import sys
import os
import cv2
from PyQt5 import uic, QtWidgets, QtGui, QtMultimedia, QtMultimediaWidgets
from PyQt5 import QtCore
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PRNet_faceswap_tool import PRNetfacetool, DEFAULT_VIDEO_OUTPUT
from cartoon import Photo2Cartoon
import numpy as np

PR = PRNetfacetool()

# The file dialogs used to start in a hardcoded Windows path ("D:\\") which
# does not exist on macOS / Linux.
HOME_DIR = os.path.expanduser('~')

# Qt gives its threads a 512 KB stack on macOS. numpy.linalg (used by
# skimage's estimate_transform -> tform.inverse) goes through OpenBLAS, whose
# LAPACK routines reserve a large buffer on the stack and blow up with
# "Bus error" on such a small stack. The main thread has 16 MB, so give the
# workers a comfortable margin.
WORKER_STACK_SIZE = 64 * 1024 * 1024

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('./Qtuic/main.ui', self)
        self.FaceImgWindow = FaceSwap()
        self.FaceVideoWindow = FaceSwapVideo()
        self.pushButton_imgswap.clicked.connect(self.FaceImgWindow.show)
        self.pushButton_videoswap.clicked.connect(self.FaceVideoWindow.show)

class FaceSwapVideo(QtWidgets.QMainWindow):
    def __init__(self):
        super(FaceSwapVideo, self).__init__()
        uic.loadUi('./Qtuic/faceswapVideo.ui', self)
        self.update()
        # self.mediaPlayer = QMediaPlayer(self)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QtMultimediaWidgets.QVideoWidget()
        # self.videoWidget.setGeometry(self.pos().x(), self.pos().y(), self.width(), self.height())
        # self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.vbox = QtWidgets.QVBoxLayout(self)
        self.vbox.addWidget(self.videoWidget)
        self.video_widget.setLayout(self.vbox)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        # self.video_widget.setVisible(False)

        self.pushButton_right.clicked.connect(self.getvideofile)
        self.pushButton_left.clicked.connect(self.getOriImg)
        self.pushButton_start.clicked.connect(self.faceswap_start)

        self.faceimg = None
        self.ori_video_path = None
        self.worker = None
        self._start_button_text = self.pushButton_start.text()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            reply = QtWidgets.QMessageBox.question(
                self, 'Window Close',
                '影片處理中，確定要關閉視窗嗎？',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.worker.stop()
            self.worker.wait()
        reply = QtWidgets.QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            print('Window closed')
            self.mediaPlayer.stop()
            self.hide()
        else:
            event.ignore()

    def faceswap_start(self):
        if self.worker is not None and self.worker.isRunning():
            # second click while running -> cancel
            self.statusbar.showMessage('正在取消...')
            self.worker.stop()
            return

        if self.faceimg is None or self.ori_video_path is None:
            QtWidgets.QMessageBox.warning(
                self, '缺少素材',
                '請先選擇「外觀」照片與要處理的影片。')
            return

        self.worker = FaceSwapVideo_thread(
            PR, self.faceimg, self.ori_video_path, DEFAULT_VIDEO_OUTPUT)
        self.worker.progress.connect(self.on_progress)
        self.worker.video_done.connect(self.on_video_done)
        self.worker.video_failed.connect(self.on_video_failed)
        self.worker.finished.connect(self.on_worker_finished)

        self.pushButton_start.setText('取消')
        self.statusbar.showMessage('影片處理中...')
        self.worker.start()

    def on_progress(self, frame_index, total_frames):
        if total_frames > 0:
            self.statusbar.showMessage(
                '影片處理中... %d / %d  (%.0f%%)' % (
                    frame_index, total_frames, 100.0 * frame_index / total_frames))
        else:
            self.statusbar.showMessage('影片處理中... %d 影格' % frame_index)

    def on_video_done(self, output_path):
        self.statusbar.showMessage('完成：%s' % output_path)
        QtWidgets.QMessageBox.information(
            self, '完成', '影片換臉完成，輸出檔案：\n%s' % output_path)
        # show the result in the preview player
        if os.path.exists(output_path):
            self.mediaPlayer.setMedia(QtMultimedia.QMediaContent(
                QtCore.QUrl.fromLocalFile(output_path)))
            self.mediaPlayer.play()

    def on_video_failed(self, message):
        self.statusbar.showMessage('失敗')
        QtWidgets.QMessageBox.critical(self, '失敗', '影片處理失敗：\n%s' % message)

    def on_worker_finished(self):
        self.pushButton_start.setText(self._start_button_text)
        self.worker = None

    def getOriImg(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',HOME_DIR,"Image files(*.jpg *.png *.jpeg)")
        if fname:
            self.faceimg = cv2.imread(fname)
            img = self.faceimg.copy()
            h, w, c = img.shape
            n = 1
            while(h > 540 or w > 540):
                n = n-0.1
                h = int (h*n)
                w = int (w*n)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            img = PR.PRNetdrawrect(img)
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.img_label.setPixmap(QtGui.QPixmap.fromImage(qImg))


    def getvideofile(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',HOME_DIR,"Video files(*.mp4 *.mov *.avi *.mkv)")
        if fname:
            self.ori_video_path = fname
            # # self.video_widget.setLayout(self.vbox)
            # # self.video_widget.setVisible(True)

            self.mediaPlayer.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(fname)))
            self.mediaPlayer.play()



class FaceSwap(QtWidgets.QMainWindow):
    def __init__(self):
        super(FaceSwap, self).__init__()
        uic.loadUi('./Qtuic/faceswap.ui', self)
        self.faceswap = FaceSwap_thread()
        self.ori_img = None
        self.ref_img = None
        self.turncartoon = Photo2Cartoon()
        self._swap_result_connected = False
        self.pushButton_left.clicked.connect(self.getimgfile)
        self.pushButton_right.clicked.connect(self.getimgfile_rightbutton)
        self.pushButton_Faceswap.clicked.connect(self.send_sth)
        
        self.pushButton_left_down.clicked.connect(self.cartoonstyle_left)
        self.pushButton_right_down.clicked.connect(self.cartoonstyle_right)
        
        
        # self.actionOpenfile.triggered.connect(self.)

    def send_sth(self):
        if self.ori_img is None or self.ref_img is None:
            QtWidgets.QMessageBox.warning(
                self, '缺少素材', '請先選擇兩張圖片再進行換臉。')
            return

        # The result signal has to be connected *before* the thread starts,
        # otherwise the worker can emit before anyone is listening and the
        # result is lost.  The connection is only made once, otherwise every
        # click would add another duplicate connection.
        if not self._swap_result_connected:
            self.faceswap.regraph_trigger.connect(self.showfaceswapped_img)
            self._swap_result_connected = True

        self.faceswap.get_ori_img(self.ori_img)
        self.faceswap.get_ref_img(self.ref_img)
        self.faceswap.start()

    def showfaceswapped_img(self, array):
        result = array
        # h, w, c = result.shape

        img = result.copy()
        h, w, c = img.shape
        while(w > 512): 
            h = int (h-1)
            w = int (w-1)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)    
        
        bytesPerline = 3*w
        qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label_mid.setPixmap(QtGui.QPixmap.fromImage(qImg))
    
    def cartoonstyle_left(self):
        if self.ori_img is not None and PR.IS_face(self.ori_img):
            self.ori_img = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2RGB)
            self.ori_img = self.turncartoon.inference(self.ori_img)
            img = self.ori_img.copy()
            img = PR.PRNetdrawrect(img)
            h, w, c = img.shape
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_left.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_left_down.hide()
        else:
            print('no face')
    def cartoonstyle_right(self):
        if self.ref_img is not None and PR.IS_face(self.ref_img):
            self.ref_img = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB)
            self.ref_img = self.turncartoon.inference(self.ref_img)
            img = self.ref_img.copy()
            img = PR.PRNetdrawrect(img)
            h, w, c = img.shape
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_right.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_right_down.hide()
        else:
            print('no face')

    def getimgfile_rightbutton(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',HOME_DIR,"Image files(*.jpg *.png *.jpeg)")
        # self.label_right.setPixmap(QtGui.QPixmap(fname))
        self.ref_img = cv2.imread(fname)
        if (fname):
            print(self.ref_img.shape)
            h, w, c = self.ref_img.shape
            while(h > 1080 or w > 1080): 
                h = int (h*0.5)
                w = int (w*0.5)
                self.ref_img = cv2.resize(self.ref_img, (w, h), interpolation=cv2.INTER_CUBIC)
            
            img = PR.PRNetdrawrect(self.ref_img)
            
            h, w, c = img.shape
            while(w > 512): 
                h = int (h-1)
                w = int (w-1)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_right.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_right_down.show()
    
    def getimgfile(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',HOME_DIR,"Image files(*.jpg *.png *.jpeg)")
        # self.label_right.setPixmap(QtGui.QPixmap(fname))
        self.ori_img = cv2.imread(fname)
        if (fname):
            print(self.ori_img.shape)
            h, w, c = self.ori_img.shape
            while(h > 1080 or w > 1080): 
                h = int (h*0.5)
                w = int (w*0.5)
                self.ori_img = cv2.resize(self.ori_img, (w, h), interpolation=cv2.INTER_CUBIC)
            
            img = PR.PRNetdrawrect(self.ori_img)

            h, w, c = img.shape
            while(w > 512): 
                h = int (h-1)
                w = int (w-1)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_left.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_left_down.show()

        # self.label_left.setScaledContents(True)

class FaceSwapVideo_thread(QtCore.QThread):
    '''Runs the (slow) video face swap off the GUI thread.'''

    progress = QtCore.pyqtSignal(int, int)   # frame index, total frames
    video_done = QtCore.pyqtSignal(str)      # output path
    video_failed = QtCore.pyqtSignal(str)    # error message

    def __init__(self, tool, faceimg, video_path, output_path):
        super(FaceSwapVideo_thread, self).__init__()
        self.setStackSize(WORKER_STACK_SIZE)
        self.tool = tool
        self.faceimg = faceimg
        self.video_path = video_path
        self.output_path = output_path
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self.tool.PRNetvideo(
                self.faceimg, self.video_path,
                output_path=self.output_path,
                progress=lambda i, t: self.progress.emit(i, t),
                should_stop=lambda: self._stop_requested)
            self.video_done.emit(self.output_path)
        except Exception as exc:  # noqa: BLE001 - report anything to the GUI
            self.video_failed.emit('%s: %s' % (type(exc).__name__, exc))

class FaceSwap_thread(QtCore.QThread):
    regraph_trigger = QtCore.pyqtSignal(object)
    def __init__(self):
        super(FaceSwap_thread, self).__init__()
        self.setStackSize(WORKER_STACK_SIZE)
        self.ori_img = None
        self.ref_img = None

    def run(self):
        self.faceswaps()
    
    

    def get_ori_img(self, array):
        self.ori_img = array

    def get_ref_img(self, array):
        self.ref_img = array
        
    def faceswaps(self):
        if(self.ori_img is not None and self.ref_img is not None):
            if PR.IS_face(self.ref_img)  and PR.IS_face(self.ori_img):
                result = PR.PRNetfaceswap(self.ref_img, self.ori_img)
                # h, w, c = result.shape
                # n = 1
                # while(h > 540 or w > 540): 
                #     n = n-0.1
                #     h = int (h*n)
                #     w = int (w*n)
                #     result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)
                self.regraph_trigger.emit(result)
            else:
                print("No face")
        else:
            print('no')


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())