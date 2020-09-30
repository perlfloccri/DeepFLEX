import numpy as np
import cv2
import os
import tifffile as tiff
import pickle
from skimage import img_as_float
# ============================================================================


FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (0, 255, 255)

# ============================================================================


class PolygonDrawer(object):
    def __init__(self, window_name, img):
        self.window_name = window_name # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self._img = img # current fluorescence image

    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 600, 600)
        cv2.imshow(self.window_name, self._img)
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while not self.done:
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            canvas = self._img.copy()
            if len(self.points) > 0:
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, thickness=3)
                # And  also show what the current segment would look like
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR, thickness=3)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finished entering the polygon points, so let's make the final drawing
        canvas = self._img
        # of a filled polygon
        if len(self.points) > 0:
            cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
        # And show it
        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)

        # Show final mask as a control
        new_img = np.zeros((self._img.shape[0], self._img.shape[1]), np.uint8)
        if len(self.points) > 0:
            cv2.fillPoly(new_img, np.array([self.points]), FINAL_LINE_COLOR)
        cv2.namedWindow('Selected Region', flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Selected Region', 600, 600)
        cv2.imshow('Selected Region', new_img)
        cv2.waitKey()
        cv2.destroyWindow(self.window_name)
        return new_img

# ============================================================================

def vis_img(img: np.ndarray):
    img_float = img_as_float(img.astype(int))
    img_float = img_float - np.percentile(img_float[20:-20, 20:-20], 0.135)  # subtract background
    if not np.percentile(img_float[20:-20, 20:-20], 100 - 0.135) == 0.0:
        img_float /= np.percentile(img_float[20:-20, 20:-20], 100 - 0.135)  # normalize to 99.865% of max value
    img_float[img_float < 0] = 0
    img_float[img_float > 1] = 1  # cut-off high intensities

    return img_float

in_path = r'G:\FORSCHUNG\LAB4\_Projects\VISIOMICS\MELC\2020'
out_path = r'G:\FORSCHUNG\LAB4\_Papers_lab4\Multiplex imaging 2020\DeepFlex\MELC_raw_data\RoI'

samples = [s for s in os.listdir(in_path) if [char for char in s][0] == '3']
names_old = ['18-6056', '19-2647', '19-4719', '19-0473', '19-1612', '19-5187', '19-2548', '19-6196']
names_new = ['BM_1_1', 'BM_1_2', 'BM_1_3', 'BM_2_1', 'BM_2_2', 'BM_2_3', 'BM_3_1', 'BM_4_1']

for s in samples:
    fovs = [f for f in os.listdir(in_path + '\\' + s) if [char for char in f][0] == '2']
    for f in fovs:
        name = names_new[names_old.index(s.split('_')[-1])] + '_' + [char for char in f][-1]
        with open(in_path + '\\' + s + '\\' + f + '\\processed\\cell_mask.pickle', 'rb') as f:
            cell_mask = pickle.load(f)
        cell_mask[cell_mask != 0] = 255
        pd = PolygonDrawer(name, vis_img(cell_mask))
        roi = pd.run()
        print("Polygon = %s" % pd.points)
        tiff.imwrite(out_path + '\\' + name + '.tiff', roi)

        with open(out_path + '\\' + name + '.pickle', 'wb') as f:
            pickle.dump(roi, f)