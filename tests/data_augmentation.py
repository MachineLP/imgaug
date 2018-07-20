# coding=utf-8

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


class DataAug:
    def __init__(self, image):
        self.img = image
        # 随机通道处理,加减100以内
        # self.aug_WithChannels = iaa.WithChannels((0,2), iaa.Add((-100, 100)))
        # 随机裁剪和填充，percent为裁剪与填充比例，负数为放大后裁剪，正数为缩小和填充，pad_mode为填充方式，pad_cval为当空白填充时，填充像素值
        self.aug_CropAndPad = iaa.CropAndPad(percent=(-0.05, 0.1),pad_mode=ia.ALL,pad_cval=(0, 255))
        # 随机水平翻转,参数为概率
        self.aug_Fliplr = iaa.Fliplr(0.5)
        # 随机垂直翻转,参数为概率
        self.aug_Flipud = iaa.Flipud(0.5)
        # 超像素表示,p_replace被超像素代替的百分比,n_segments分割块数
        self.aug_Superpixels = iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))
        # 灰度化 (0.0,1.0),前者为偏彩色部分，后者为偏灰度部分，随机灰度化
        self.aug_GrayScale = iaa.Grayscale(alpha=(0.0, 0.6))
        # 高斯模糊
        self.aug_GaussianBlur = iaa.GaussianBlur(sigma=(0, 3.0))
        # 均值模糊，k为kernel size
        self.aug_AverageBlur = iaa.AverageBlur(k=(2, 7))
        # 中值模糊, k为kernel size
        self.aug_MedianBlur = iaa.MedianBlur(k=(3, 11))
        # 双边滤波,d为kernel size,sigma_color为颜色域标准差,sigma_space为空间域标准差
        self.aug_BilateralBlur = iaa.BilateralBlur(sigma_color=(0, 250), sigma_space=(0, 250), d=(3, 7))
        # 锐化
        self.aug_Sharpen = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
        # 浮雕效果
        self.aug_Emboss = iaa.Emboss(alpha=(0.0, 1.0), strength=(0.0, 1.5))
        # 边缘检测
        self.aug_EdgeDetect = iaa.EdgeDetect(alpha=(0.0, 1.0))
        # 方向性边缘检测
        self.aug_DirectedEdgeDetece = iaa.DirectedEdgeDetect(alpha=(0.0, 1.0), direction=(0.0, 1.0))
        # 暴力叠加像素值,每个像素统一加一个值
        self.aug_Add =  iaa.Add((-40, 40))
        # 暴力叠加像素值，每个像素加不同的值
        self.aug_AddElementwise = iaa.AddElementwise((-40, 40))
        # 随机高斯加性噪声
        self.aug_AdditiveGaussianNoise =iaa.AdditiveGaussianNoise(scale=(0.0,0.1 * 255))
        # 暴力乘法,每个像素统一乘以一个值
        self.aug_Multiply = iaa.Multiply((0.8, 1.2))
        # 暴力乘法,每个像素乘以不同值
        self.aug_MultiplyElementwise = iaa.MultiplyElementwise((0.8, 1.2))
        # 随机dropout像素值
        self.aug_Dropout = iaa.Dropout(p=(0, 0.2))
        # 随机粗dropout,2*2方块像素被dropout
        self.aug_CoarseDropout = iaa.CoarseDropout(0.02, size_percent=0.5)
        # 50%的图片,p概率反转颜色
        self.aug_Invert = iaa.Invert(0.25, per_channel=0.5)
        # 对比度归一化
        self.aug_ContrastNormalization = iaa.ContrastNormalization((0.5, 1.5))
        # 仿射变换
        self.aug_Affine = iaa.Affine(rotate=(0,20),scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
        # 仿射变换, 局部像素仿射扭曲
        self.aug_PiecewiseAffine = iaa.PiecewiseAffine(scale=(0.01, 0.05))
        # 单应性变换
        self.aug_PerspectiveTransform = iaa.PerspectiveTransform(scale=(0.01, 0.1))
        # 弹性变换
        self.aug_ElasticTransformation = iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
        # 简单的加噪,小黑块
        self.aug_SimplexNoiseAlpha =  iaa.SimplexNoiseAlpha(iaa.OneOf([iaa.EdgeDetect(alpha=(0.0, 0.5)),iaa.DirectedEdgeDetect(alpha=(0.0, 0.5), direction=(0.0, 1.0)),]))
        # 频域加噪，表现为色彩的块状变换
        self.aug_FrequencyNoiseAlpha = iaa.FrequencyNoiseAlpha(exponent=(-4, 0),first=iaa.Multiply((0.5, 1.5), per_channel=True),second=iaa.ContrastNormalization((0.5, 2.0)))

    # 示例
    def run(self):
        method_list = [self.aug_Fliplr,self.aug_Flipud,self.aug_GrayScale,self.aug_BilateralBlur,
                       self.aug_Sharpen, self.aug_Add,self.aug_Multiply,self.aug_CoarseDropout,self.aug_ContrastNormalization,
                       self.aug_AdditiveGaussianNoise,self.aug_PerspectiveTransform,self.aug_Affine,
                       self.aug_FrequencyNoiseAlpha]

        # 随机采用method里的0-method_num种方法
        method_num = 2 #len(method_list)
        processor = iaa.SomeOf((0,method_num),method_list)
        return processor.augment_images(self.img)

    def self_define_function_1(self):
        method_list = []

        # 随机采用method里的0-method_num种方法
        method_num = 2  # len(method_list)
        processor = iaa.SomeOf((0, method_num), method_list)
        return processor.augment_images(self.img)

if __name__ == '__main__':
    import cv2
    img = cv2.imread('lp.jpg')
    print (img)
    img = DataAug([img]).run()
    print (img)




