import numpy as np
import cv2
from os.path import join
import yaml
import argparse
from real_robots.omnirobot_utils.utils import PosTransformer
from scipy.spatial.transform import Rotation as R
from real_robots.constants import *
import pdb

def rotateMatrix90(matrix):
    """

    :param matrix: (array) matrix to rotate of 90 degrees
    :return: (array)
    """
    new_matrix = np.transpose(matrix)
    new_matrix = np.flip(new_matrix, axis=1)
    return new_matrix


class SquareFinder():
    """
    Find the exact physical position of marker in the coordinate of camera.
    """
    def __init__(self, camera_info_path):
        """
        :param camera_info_path: (str)
        """
        self.min_area = 300
        with open(camera_info_path, 'r') as stream:
            try:
                contents = yaml.load(stream)
                camera_matrix = np.array(contents['camera_matrix']['data'])
                self.origin_size = np.array([contents['image_width'], contents['image_height']])
                self.camera_matrix = np.reshape(camera_matrix, (3, 3))
                self.distortion_coefficients = np.array(contents['distortion_coefficients']['data'])
            except yaml.YAMLError as exc:
                print(exc)
        self.img = None
        self.last_target_pos = [0, 0]
        self.channel = None

    def setChannel(self, channel):
        self.channel = channel

    def intersection(self, l1, l2):
        """
        calculate the intersection point of two lines
        :param l1:
        :param l2:
        :return:
        """
        vx = l1[0]
        vy = l1[1]
        ux = l2[0]
        uy = l2[1]
        wx = l2[2] - l1[2]
        wy = l2[3] - l1[3]

        tmp = vx * uy - vy * ux
        if tmp == 0:
            tmp = 1

        s = (vy * wx - vx * wy) / tmp
        px = l2[2] + s * ux
        py = l2[3] + s * uy

        return px, py

    def labelSquares(self, img, visualise):
        """
        label all the candidate squares
        :param img:
        :param visualise: (bool)
        :return:
        """
        self.img = img
        self.gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        self.edge = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5)

        # changed in opencv version 4.*.* where there is only 2 outputs
        # cnts, _ = cv2.findContours(self.edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        _, cnts, _ = cv2.findContours(self.edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        candidate_contours = []
        candidate_approx = []

        for contour in cnts:
            if len(contour) < 50:  # filter the short contour
                continue
            approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.035, True)
            l1 = np.linalg.norm(approx[1] - approx[0])
            l2 = np.linalg.norm(approx[2] - approx[1])
            if len(approx) == 4 and min(l1, l2) > 20 and abs(cv2.contourArea(approx)) > self.min_area \
                    and cv2.isContourConvex(approx):
                cv2.drawContours(img, approx, -1, (0, 255, 0), 3)
                candidate_approx.append(approx)
                candidate_contours.append(contour)

        n_blob = len(candidate_approx)
        self.blob_corners = np.zeros((n_blob, 4, 2), np.float32)

        for i in range(n_blob):
            # find how much points are in each line (length)
            fitted_lines = np.zeros((4, 4), np.float)
            for j in range(4):
                pt0 = candidate_approx[i][j]
                pt1 = candidate_approx[i][(j + 1) % 4]
                k0 = -1
                k1 = -1
                # find corresponding approximated point (pt0, pt1) in contours
                for k in range(len(candidate_contours[i])):
                    pt2 = candidate_contours[i][k]
                    if pt2[0][0] == pt0[0][0] and pt2[0][1] == pt0[0][1]:
                        k0 = k
                    if pt2[0][0] == pt1[0][0] and pt2[0][1] == pt1[0][1]:
                        k1 = k

                # compute how much points are in this line
                if k1 >= k0:
                    length = k1 - k0 - 1
                else:
                    length = len(candidate_contours[i]) - k0 + k1 - 1

                if length == 0:
                    length = 1

                line_pts = np.zeros((1, length, 2), np.float32)
                # append this line's point to array 'line_pts'
                for l in range(length):
                    ll = (k0 + l + 1) % len(candidate_contours[i])
                    line_pts[0, l, :] = candidate_contours[i][ll]

                # Fit edge and put to vector of edges
                [vx, vy, x, y] = cv2.fitLine(line_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                fitted_lines[j, :] = [vx, vy, x, y]
                if visualise:
                    # Finally draw the line
                    # Now find two extreme points on the line to draw line
                    left = [0, 0]
                    right = [0, 0]
                    length = 100
                    left[0] = x - vx * length
                    left[1] = y - vy * length
                    right[0] = x + vx * length
                    right[1] = y + vy * length
                    cv2.line(img, tuple(left), tuple(right), 255, 2)

            # Calculated four intersection points
            for j in range(4):
                intc = self.intersection(fitted_lines[j, :], fitted_lines[(j + 1) % 4, :])
                self.blob_corners[i, j, :] = intc

            if visualise:
                for j in range(4):
                    intc = tuple(self.blob_corners[i, j, :])
                    if j == 0:
                        cv2.circle(img, intc, 5, (255, 255, 255))
                    if j == 1:
                        cv2.circle(img, intc, 5, (255, 0, 0))
                    if j == 2:
                        cv2.circle(img, intc, 5, (0, 255, 0))
                    if j == 3:
                        cv2.circle(img, intc, 5, (0, 0, 255))
                cv2.imshow('frame', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def getReward(self, i_square):
        self.img_detected = self.img[:, :, self.channel]
        reward = 0
        for i in range(self.gray.shape[0]):
            for j in range(self.gray.shape[1]):
                reward += self.marker_mask[i, j] * self.img_detected[i, j]
        reward = reward / cv2.contourArea(self.blob_corners[i_square, :, :])
        return reward

    def findSquare(self, visualise=False):
        """
        TODO find the square's pixel position
        :param marker_id:
        :param visualise: (bool)
        :return:
        """

        # print("find marker_id: ", marker_id)
        reward = []
        for i_square in range(self.blob_corners.shape[0]):
            # create marker's mask
            self.marker_mask = np.zeros(self.gray.shape[0: 3], np.uint8)
            cv2.fillConvexPoly(self.marker_mask,
                               self.blob_corners[i_square, :, :].astype(np.int32).reshape(-1, 1, 2), 1)
            reward.append(self.getReward(self.img, i_square))
        index = reward.index(max(reward))
        if reward[index] > 200:
            target_pos = np.mean(self.blob_corners[index, :, :])
            self.last_target_pos = target_pos
            return target_pos
        return self.last_target_pos

    def getSquarePose(self, img, visualise=False):
        """
        TODO
        :param img:
        :param marker_ids:
        :param visualise: (bool)
        :return:
        """
        self.labelSquares(img, visualise)
        return self.findSquare(visualise)

if __name__ == "__main__":
    camera_info_path = "real_robots/omnirobot_utils/cam_calib_info.yaml"
    with open(camera_info_path, 'r') as stream:
        try:
            contents = yaml.load(stream)
            camera_matrix = np.array(contents['camera_matrix']['data'])
            origin_size = np.array(
                [contents['image_height'], contents['image_width']])
            camera_matrix = np.reshape(camera_matrix, (3, 3))
            dist_coeffs = np.array(
                contents["distortion_coefficients"]["data"]).reshape((1, 5))
        except yaml.YAMLError as exc:
            print(exc)
    cropped_img = cv2.imread("real_robots/omnirobot_utils/image.png")
    # cap = cv2.VideoCapture(0)

    img = np.zeros((480, 640, 3), np.uint8)
    img[0:480, 80:560, :] = cv2.resize(cropped_img, (480, 480))
    # ret, frame = cap.read()
    square_finder = SquareFinder(camera_info_path)
    square_finder.setChannel(2)
    pos_coord_pixel = square_finder.getSquarePose(img)
    print("square position : ", pos_coord_pixel)
    cv2.circle(img, tuple(pos_coord_pixel),5,[0,255,0],thickness=5)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # while (True):
    #     ret, frame = cap.read()
    #
    #     pos_coord_pixel = maker_finder.getMarkerPose(frame, 'robot', visualise=False)
    #     print("position in the image: ", pos_coord_pixel)
    #
    #     cv2.circle(frame, tuple(pos_coord_pixel), 5, [0,255,0],thickness=5)
    #     cv2.imshow('test', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
