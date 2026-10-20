import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import render as rr
from threeD_utilsv2 import threetool

DEFAULT_VIDEO_OUTPUT = os.path.join('videos', 'PRNetvideo.mov')


class ReferenceFace:
    '''3D reconstruction of the face that is going to be pasted onto the target.

    The reconstruction only depends on the reference picture, so it can be
    computed once and reused for every frame of a video.
    '''

    def __init__(self, tool, ref_img):
        self.pos, self.vertices, self.center = tool.tduv2t.pre_v3_posandvertice(ref_img)
        self.texture = tool.tduv2t.get_texture(ref_img / 255., self.pos)


class PRNetfacetool:
    def __init__(self, shading='gouraud'):
        self.tduv2t = threetool()
        self._eye_mask = None
        # 'gouraud' interpolates the vertex colours inside every triangle,
        # 'flat' reproduces the original implementation (one average colour
        # per triangle) bit for bit.
        self.shading = shading

    def IS_face(self, img):
        IS_detefaced = self.tduv2t.deteface(img)
        if len(IS_detefaced) == 0 :
            return False
        else:
            return True

    def PRNetdrawrect(self, img):
        
        IS_detefaced = self.tduv2t.deteface(img)
        if len(IS_detefaced) == 0 :
            return img
        else:
            pos = self.tduv2t.pre_v3_posandvertice(img, get_vertices_center=False)
            face_kps = self.tduv2t.face_kps(pos)
            face_kps_arry = []
            for i in range(face_kps.shape[0]):
                x = face_kps[i, 0]
                y = face_kps[i, 1]
                face_kps_arry.append((x, y))
            points = np.array(face_kps_arry, np.int32)
            x, y, w, h = cv2.boundingRect(points)
            face = cv2.rectangle(img.copy(), (x, y), (x+w,y+h), (0, 255, 0), 2)
            return face


        
    def blendImages(self, src, dst, mask, featherAmount=0.2):
        #indeksy nie czarnych pikseli maski
        maskIndices = np.where(mask != 0)
        #te same indeksy tylko, ze teraz w jednej macierzy, gdzie kazdy wiersz to jeden piksel (x, y)
        maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
        faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
        featherAmount = featherAmount * np.max(faceSize)

        hull = cv2.convexHull(maskPts)
        dists = np.zeros(maskPts.shape[0])
        for i in range(maskPts.shape[0]):
            # OpenCV >= 4.x rejects numpy scalars here; pass plain floats.
            dists[i] = cv2.pointPolygonTest(
                hull, (float(maskPts[i, 0]), float(maskPts[i, 1])), True)

        weights = np.clip(dists / featherAmount, 0, 1)

        composedImg = np.copy(dst)
        composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

        return composedImg
###############################################################
    def get_eye_mask(self):
        '''Eye region mask of the UV template (loaded from disk only once).'''
        if self._eye_mask is None:
            uv_face_eye = cv2.imread('./Data/uv-data/uv_face_eyes.png', cv2.IMREAD_GRAYSCALE) / 255
            uv_face = cv2.imread('./Data/uv-data/uv_face.png', cv2.IMREAD_GRAYSCALE) / 255
            self._eye_mask = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)
        return self._eye_mask

    def build_reference(self, ref_img):
        '''Pre-compute the 3D reconstruction of the reference face.

        Reuse the returned object for every frame of a video, the reference
        face does not change from frame to frame.
        '''
        return ReferenceFace(self, ref_img)

    def PRNetvideo(self, img, video_path, output_path=None, progress=None,
                   should_stop=None, preview=None):
        '''Swap the face of `img` onto every frame of `video_path`.

        Args:
            img: the reference face (BGR).
            video_path: input video.
            output_path: output video, defaults to ./videos/PRNetvideo.mov
            progress: optional callback(frame_index, total_frames).
            should_stop: optional callback() -> bool, return True to abort.
            preview: optional callback(frame) to display the processed frame.
        '''
        if output_path is None:
            output_path = DEFAULT_VIDEO_OUTPUT
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError('cannot open video: %s' % video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 20.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outPRNet = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # The reference face is reconstructed once and reused for every frame.
        reference = self.build_reference(img)

        print('start: %d frames @ %dx%d %.1f fps' % (total_frames, width, height, fps))
        processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if width and height and (frame.shape[1] != width or frame.shape[0] != height):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

                if self.IS_face(frame):
                    frame = self.PRNetfaceswap(frame, reference=reference)
                outPRNet.write(frame)
                processed += 1

                if preview is not None:
                    preview(frame)
                if progress is not None:
                    progress(processed, total_frames)
                if should_stop is not None and should_stop():
                    print('stopped by user after %d frames' % processed)
                    break
        finally:
            cap.release()
            outPRNet.release()

        print('Face Swap Finished -> %s (%d frames)' % (output_path, processed))
        return output_path
###############################################################
    def PRNetfaceswap(self, img, ref_img=None, reference=None):
        # first person
        imgdtype = img.dtype
        [h, w, _] = img.shape
        # 3d reconstruction -> get texture
        pos, vertices,center = self.tduv2t.pre_v3_posandvertice(img)
        img = img/255
        texture = self.tduv2t.get_texture(img,pos)
        center = center.astype(int)
        center =tuple(center)
        # Second person (reconstruction is reused when a reference is given)
        if reference is None:
            if ref_img is None:
                raise ValueError('either ref_img or reference must be given')
            reference = self.build_reference(ref_img)
        ref_texture = reference.texture
        # load eye mask
        eye_mask = self.get_eye_mask()
        # new_texture
        new_texture = self.blendImages(ref_texture*255,texture*255,eye_mask)/255.0

        ##
        triangles = self.tduv2t.get_triangles()

        vis_colors = np.ones((vertices.shape[0], 1))
        face_mask = rr.render_texture(vertices.T, vis_colors.T, triangles.T, h, w, c = 1,
                                      shading=self.shading)
        face_mask = np.squeeze(face_mask > 0).astype(np.float32)


        new_colors = self.tduv2t.get_colors_from_texture(new_texture)
        new_img = rr.render_texture(vertices.T, new_colors.T, triangles.T, h, w, c = 3,
                                   shading=self.shading)

        new_image = img*(1 - face_mask[:,:,np.newaxis]) + new_img*face_mask[:,:,np.newaxis]

        new_image = new_image*255
        new_image = new_image.astype(imgdtype)
        # NOTE: the original code also called cv2.seamlessClone() here but threw
        # the result away, which only wasted time (and could raise on a
        # degenerate mask), so the call has been dropped.
        return new_image


if __name__ == '__main__':
    img = cv2.imread("./images/photo_test.jpg")
    ref_img = cv2.imread("./images/face04.jpg")
    faceswap = PRNetfacetool()
    cv2.imshow('result', faceswap.PRNetfaceswap(img, ref_img))
    cv2.waitKey(0)
    