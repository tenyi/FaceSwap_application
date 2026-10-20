'''Face detection + cartoonisation over a whole video.

Usage:
    python UseVideo.py <input video> [--out-dir videos]

Writes two files next to the input:
    <name>_detected.mov   the original video with a face box drawn on every frame
    <name>_cartoon.mov    the cartoon version of every 10th frame
'''
import argparse
import os
import sys

import cv2
import numpy as np

from threeD_utilsv2 import threetool
from cartoon import Photo2Cartoon


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', help='input video file')
    parser.add_argument('--out-dir', default='videos', help='output directory')
    parser.add_argument('--cartoon-every', type=int, default=10,
                        help='run the cartoon model on every Nth frame')
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        print('video not found: %s' % args.video)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    detected_path = os.path.join(args.out_dir, stem + '_detected.mov')
    cartoon_path = os.path.join(args.out_dir, stem + '_cartoon.mov')

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print('cannot open video: %s' % args.video)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(detected_path, fourcc, fps, (width, height))
    outcartoon = cv2.VideoWriter(cartoon_path, fourcc, fps, (256, 256))

    c2p = Photo2Cartoon()
    tduv2t = threetool()

    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            resframe = cv2.resize(frame, (width, height))
            faceframe = resframe.copy()

            if count % max(1, args.cartoon_every) == 0:
                imgg = cv2.cvtColor(resframe, cv2.COLOR_BGR2RGB)
                cartoon = c2p.inference(imgg)
                if cartoon is not None:
                    outcartoon.write(cartoon)

            for point in tduv2t.deteface(faceframe):
                x1, y1 = point[0], point[1]
                x2, y2 = point[0] + point[2], point[1] + point[3]
                cv2.rectangle(faceframe, (x1, y1), (x2, y2), (0, 255, 0), 3)
            out.write(faceframe)
            count += 1
    finally:
        cap.release()
        out.release()
        outcartoon.release()

    print('%d frames processed' % count)
    print('detected video : %s' % detected_path)
    print('cartoon video  : %s' % cartoon_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
