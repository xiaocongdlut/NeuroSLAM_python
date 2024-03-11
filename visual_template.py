# -*- coding: utf-8 -*-
# @Time : 2023/3/8 15:58
# @Author : xiao cong
# @Description :

import cv2
import numpy as np

class ViewTemplate(object):

    def __init__(self, gcX, gcY, gcZ, curYawTheta, curPitchTheta):   # curPitchTheta 就是高度？
        self.gcX = gcX
        self.gcY = gcY
        self.gcZ = gcZ
        self.curYawTheta = curYawTheta
        self.curPitchTheta = curPitchTheta
        self.exps = []

    # def match(self, input_frame):
    #     raise NotImplemented


class IntensityProfileTemplate(ViewTemplate):
    """A very simple view template as described in Milford and Wyeth's
       original algorithm.  Basically, just compute an intensity profile
       over some region of the image. Simple, but suprisingly effective
       under certain conditions

    """

    def __init__(self, input_frame, gcX, gcY, gcZ, curYawTheta, curPitchTheta,
                 VT_IMG_CROP_Y_RANGE, VT_IMG_CROP_X_RANGE):
        ViewTemplate.__init__(self, gcX, gcY, gcZ, curYawTheta, curPitchTheta)

        self.VT_IMG_CROP_X_RANGE = VT_IMG_CROP_X_RANGE
        self.VT_IMG_CROP_Y_RANGE = VT_IMG_CROP_Y_RANGE
        self.VT_IMG_RESIZE_Y_RANGE = 12
        self.VT_IMG_RESIZE_X_RANGE = 16
        self.VT_IMG_HALF_OFFSET = [0, int(np.floor(self.VT_IMG_RESIZE_X_RANGE/2))]    # VT_IMG_HALF_OFFSET = [0 floor(VT_IMG_RESIZE_X_RANGE / 2)]
        self.PATCH_SIZE_Y_K = 5
        self.PATCH_SIZE_X_K = 5
        self.VT_IMG_Y_SHIFT = 3
        self.VT_IMG_X_SHIFT = 5

        self.first = True

        # compute template from input_frame
        self.template = self.convert_frame(input_frame)


    def convert_frame(self, input_frame):
        "Convert an input frame into an intensity line-profile"

        sub_im = input_frame[self.VT_IMG_CROP_Y_RANGE, self.VT_IMG_CROP_X_RANGE]
        vtResizedImg = cv2.resize(sub_im, (self.VT_IMG_RESIZE_X_RANGE, self.VT_IMG_RESIZE_Y_RANGE))  # 裁剪图片
        normVtImg = np.zeros((self.VT_IMG_RESIZE_Y_RANGE, self.VT_IMG_RESIZE_X_RANGE))

        extVtImg = np.zeros((self.VT_IMG_RESIZE_Y_RANGE + self.PATCH_SIZE_Y_K - 1, self.VT_IMG_RESIZE_X_RANGE + self.PATCH_SIZE_X_K - 1))
        extVtImg[int(np.fix((self.PATCH_SIZE_Y_K + 1) / 2) - 1): int(np.fix((self.PATCH_SIZE_Y_K + 1) / 2) \
            + self.VT_IMG_RESIZE_Y_RANGE - 1), int(np.fix((self.PATCH_SIZE_X_K + 1) / 2) - 1): int(np.fix((self.PATCH_SIZE_X_K + 1) / 2) \
                                                           + self.VT_IMG_RESIZE_X_RANGE - 1)] = vtResizedImg

        for v in range(0, self.VT_IMG_RESIZE_Y_RANGE):
            for u in range(0, self.VT_IMG_RESIZE_X_RANGE):
                patchImg = extVtImg[v: v + self.PATCH_SIZE_Y_K - 1, u: u + self.PATCH_SIZE_X_K - 1]
                meanPatchImg = np.mean(patchImg)
                stdPatchIMG = np.std(patchImg)
                normVtImg[v, u] = (vtResizedImg[v, u] - meanPatchImg) / stdPatchIMG / 255

        return normVtImg

    def compare_segments(self, seg1, seg2, slenY, slenX, cwlY, cwlX ):
        mindiff = 1e7
        minoffsetX = 1e7
        minoffsetY = 1e7
        for halfOffset in self.VT_IMG_HALF_OFFSET:
            seg2 = np.roll(seg2, halfOffset, axis=1)

            for offsetY in range(0, slenY+1):
                for offsetX in range(0, slenX+1):
                    cdiff = abs(seg1[offsetY: cwlY, offsetX: cwlX] - seg2[0: cwlY - offsetY, 0: cwlX - offsetX])
                    cdiff = sum(sum(cdiff)) / (cwlY - offsetY) * (cwlX - offsetX)
                    if cdiff < mindiff:
                        mindiff = cdiff
                        minoffsetX = offsetX
                        minoffsetY = offsetY

            for offsetY in range(1, slenY+1):
                for offsetX in range(1, slenX+1):
                    cdiff = abs(seg1[0: cwlY - offsetY, 0: cwlX - offsetX] - seg2[offsetY: cwlY, offsetX: cwlX])
                    cdiff = sum(sum(cdiff)) / (cwlY - offsetY) * (cwlX - offsetX)
                    if cdiff < mindiff:
                        mindiff = cdiff
                        minoffsetX = -offsetX
                        minoffsetY = -offsetY

        offsetX = minoffsetX
        offsetY = minoffsetY
        sdif = mindiff

        return offsetY, offsetX, sdif

    def match(self, input_frame):
        "Return a score for how well the template matches an input frame"
        normVtImg = self.convert_frame(input_frame)

        offsetY, offsetX, sdif = self.compare_segments(normVtImg,
                                          self.template,
                                          self.VT_IMG_Y_SHIFT,
                                          self.VT_IMG_X_SHIFT,
                                          normVtImg.shape[0], normVtImg.shape[1])

        return sdif


class ViewTemplateCollection(object):
    """ A collection of simple visual templates against which an incoming
        frame can be compared.  The entire collection of templates is matched
        against an incoming frame, and either the best match is returned, or
        a new template is created from the input image in the event that none
        of the templates match well enough

        (Matlab equivalent: rs_visual_template)

    """

    def __init__(self, template_generator, match_threshold, global_decay):
        """
        Arguments:
        template_generator -- a callable object that generates a ViewTemplate
                              subclass
        match_threshold -- the threshold below which a subtemplate's match
                           is considered "good enough".  Failure to match at
                           this threshold will result in the generation of a
                           new template object
        global_decay -- not currently used

        """
        self.template_generator = template_generator
        self.global_decay = global_decay
        self.match_threshold = match_threshold

        self.templates = []
        self.current_template = None

    def __getitem__(self, index):
        return self.templates[index]

    def match(self, input_frame, x, y, z, yaw, height):
        match_scores = [t.match(input_frame) for t in self.templates]
        print(len(self.templates))
        if len(match_scores) == 0 or min(match_scores) > self.match_threshold:
            # no matches, so build a new one
            new_template = self.template_generator(input_frame, x, y, z, yaw, height)
            self.templates.append(new_template)
            if len(match_scores) != 0:
                print(min(match_scores))
            return new_template

        best_template = self.templates[match_scores.index(min(match_scores))]
        print(best_template)

        return best_template

