import re
import time

import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtWidgets, QtCore

import sys

from matplotlib import pyplot as plt
from skimage.io import imread

from src.matching_algorithm.semi_global_matching import SemiGlobalMatching
from src.matching_algorithm.winner_takes_it_all import WinnerTakesItAll
from src.matching_cost.normalised_cross_correlation import NormalisedCrossCorrelation
from src.matching_cost.sum_of_absolute_differences import SumOfAbsoluteDifferences
from src.matching_cost.sum_of_squared_differences import SumOfSquaredDifferences
from src.stereo_matching import StereoMatching
from src.utilities import AccX, IO


def read_pfm(file):
    """
    读取PFM格式的图像
    """
    with open(file, "rb") as f:
        # 读取头部信息
        header = f.readline().rstrip()
        color = True if header == b'PF' else False

        # 读取尺寸信息
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('ascii'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        # 读取比例因子
        scale = float(f.readline().rstrip())
        if scale < 0:  # 如果比例因子为负，则数据为小端字节序
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # 大端字节序

        # 读取图像数据
        data = np.fromfile(f, endian + 'f')  # 根据头部信息读取浮点数
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)  # PFM格式存储的图像通常是上下颠倒的

        return data


def import_image(scenario_name):
    """
    读取左右视图
    """
    if scenario_name == "bowling" or scenario_name == "cones" or scenario_name == "Adirondack":
        # Read left and right images
        left_image = IO.import_image("./data/" + scenario_name + "_left.png")
        right_image = IO.import_image("./data/" + scenario_name + "_right.png")

        # Try to read the ground-truth and mask
        try:
            groundtruth_image = imread("./data/" + scenario_name + "_gt.png")
            mask_image = imread("./data/" + scenario_name + "_mask.png")
            if mask_image.ndim == 3:
                mask_image = mask_image[:, :, 0]
        except:
            groundtruth_image = None
            mask_image = None
            print("No ground truth found!")
    else:
        # Read left and right images
        left_image = IO.import_image("./data/" + scenario_name + "/im0.png")
        right_image = IO.import_image("./data/" + scenario_name + "/im1.png")

        # Try to read the ground-truth and mask
        try:
            groundtruth_image = read_pfm("./data/" + scenario_name + "/disp0.pfm")
            mask_image = read_pfm("./data/" + scenario_name + "/disp1.pfm")
        except:
            groundtruth_image = None
            mask_image = None
            print("No ground truth found!")

    return left_image, right_image, groundtruth_image, mask_image


def plot_image(scenario_name, left_image, right_image, groundtruth_image, mask_image):
    """
    绘制左右视图、groundtruth、mask
    """
    # Plot input images
    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(left_image, cmap='gray')
    ax1.set_title('Left', fontsize=20)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Right', fontsize=20)
    ax2.imshow(right_image, cmap='gray')

    # Try to plot the ground-truth and mask
    if groundtruth_image is not None:
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(groundtruth_image, cmap='gray')
        ax3.set_xlabel('Ground truth (left)', fontsize=20)
    else:
        print("No ground truth found!")

    if mask_image is not None:
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(mask_image, cmap='gray')
        if scenario_name == "bowling" or scenario_name == "cones" or scenario_name == "Adirondack":
            ax4.set_xlabel('Mask (left)', fontsize=20)
        else:
            ax4.set_xlabel('Ground truth (right)', fontsize=20)

    plt.savefig(f"./results/{scenario_name}_input.png")


def compute(left_image: np.ndarray, right_image: np.ndarray,
            matching_cost=SumOfAbsoluteDifferences, matching_algorithm=WinnerTakesItAll,
            groundtruth_image=None, mask_image=None,
            max_disparity=60, filter_radius=3, accx_threshold=3):
    # Convenience function for repetitive computation of images and displaying them
    #   @param[in] left_image: The left stereo image (H,W)
    #   @param[in] right_image: The right stereo image (H,W)
    #   @param[in] matching_cost: The class implementing the matching cost
    #   @param[in] matching_algorithm: The class implementing the matching algorithm
    #   @param[in] max_disparity: The maximum disparity to consider
    #   @param[in] filter_radius: The radius of the filter
    #   @param[in] groundtruth_image: Ground-truth
    #   @param[in] mask_image: The mask for excluding invalid pixels such as occluded areas
    #   @param[in] accx_threshold: Threshold disparity measure for accX accuracy measure (X)

    sm = StereoMatching(left_image, right_image, matching_cost, matching_algorithm, max_disparity, filter_radius)
    sm.compute()
    res_image = sm.result()

    try:
        accx = AccX.compute(res_image, groundtruth_image, mask_image, accx_threshold)
        print("AccX measure: " + str(accx))
    except:
        accx = None
        print("No ground truth found!")

    return res_image, accx


def run_stereo_matching(scenario_name, matching_algorithm_name, matching_cost_name):
    """
    运行双视图立体匹配
    """
    print("Picture_name: ", scenario_name)
    print("matching_algorithm: ", matching_algorithm_name)
    print("matching_cost: ", matching_cost_name)

    # Load input images
    left_image, right_image, groundtruth_image, mask_image = import_image(scenario_name)

    # Set-up algorithm
    matching_algorithm = None
    if matching_algorithm_name == "SGM":
        matching_algorithm = SemiGlobalMatching
    elif matching_algorithm_name == "WTA":
        matching_algorithm = WinnerTakesItAll
    else:
        raise ValueError("Matching algorithm '" + matching_algorithm_name + "' not recognised!")

    matching_cost = None
    if matching_cost_name == "NCC":
        matching_cost = NormalisedCrossCorrelation
    elif matching_cost_name == "SAD":
        matching_cost = SumOfAbsoluteDifferences
    elif matching_cost_name == "SSD":
        matching_cost = SumOfSquaredDifferences
    else:
        raise ValueError("Matching cost '" + matching_cost_name + "' not recognised!")

    # Perform stereo matching
    if (scenario_name != "bowling" and scenario_name != "cones" and scenario_name != "Adirondack"
            and matching_cost_name == "NCC"):
        res_image = None
        accx = None
    else:
        res_image, accx = compute(left_image, right_image, matching_cost, matching_algorithm,
                                  groundtruth_image, mask_image)

    if res_image is None:
        return res_image, accx

    # Plot result
    plt.figure()
    plt.imshow(res_image, cmap='gray')
    plt.title(f"{scenario_name}_{matching_algorithm_name}_{matching_cost_name}", fontsize=25)
    plt.savefig(f"./results/{scenario_name}_{matching_algorithm_name}_{matching_cost_name}.png")

    return res_image, accx


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.text = None
        self.graphicsView_2 = None
        self.graphicsView = None
        self.run_button = None
        self.matching_cost = None
        self.matching_algorithm = None
        self.scenario_name = None
        self.ui = None
        self.init_ui(self)  # 初始化设计的界面

    def init_ui(self, self1):
        self.ui = uic.loadUi("./Stereo_matching.ui", self1)  # 加载 .ui 文件

        self.scenario_name = self.ui.comboBox.currentText()  # 双视图
        self.matching_algorithm = self.ui.comboBox_5.currentText()  # 匹配算法
        self.matching_cost = self.ui.comboBox_6.currentText()  # 匹配cost

        self.graphicsView = self.ui.graphicsView  # 左右视图
        self.graphicsView_2 = self.ui.graphicsView_2  # 结果

        self.text = self.ui.textBrowser  # 显示结果

        # 绘制当前选择的双视图
        # self.draw_current_scenario()
        self.ui.comboBox.currentIndexChanged.connect(self.draw_current_scenario)

        # 给RUN按钮绑定事件
        run_button = self.ui.pushButton
        run_button.clicked.connect(self.run)

    def draw_current_scenario(self):
        self.text.append("Drawing pictures...")
        # 获取当前的双视图
        self.scenario_name = self.ui.comboBox.currentText()

        # 读取左右视图
        left_image, right_image, groundtruth_image, mask_image = import_image(self.scenario_name)
        # 绘制左右视图、groundtruth、mask
        plot_image(self.scenario_name, left_image, right_image, groundtruth_image, mask_image)

        self.text.append("The picture's size: " + str(left_image.shape))
        self.text.append("Click RUN to start.\n")

        self.picture_visualization()

    def picture_visualization(self):
        pixmap = QtGui.QPixmap(f"./results/{self.scenario_name}_input.png")

        # 创建 QGraphicsScene
        scene = QtWidgets.QGraphicsScene(self)
        scene.addPixmap(pixmap)
        # 创建 QRectF 对象
        rect = QtCore.QRectF(pixmap.rect())
        scene.setSceneRect(rect)

        # 设置 QGraphicsView
        self.graphicsView.setScene(scene)

        # 调整视图以适应场景的内容
        self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def result_display(self):
        pixmap = QtGui.QPixmap(f"./results/{self.scenario_name}_{self.matching_algorithm}_{self.matching_cost}.png")

        # 创建 QGraphicsScene
        scene = QtWidgets.QGraphicsScene(self)
        scene.addPixmap(pixmap)
        # 创建 QRectF 对象
        rect = QtCore.QRectF(pixmap.rect())
        scene.setSceneRect(rect)

        # 设置 QGraphicsView
        self.graphicsView_2.setScene(scene)

        # 调整视图以适应场景的内容
        self.graphicsView_2.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def run(self):
        # 获取当前的双视图、匹配算法、匹配cost
        self.scenario_name = self.ui.comboBox.currentText()
        self.matching_algorithm = self.ui.comboBox_5.currentText()
        self.matching_cost = self.ui.comboBox_6.currentText()

        # 调用main.py中的run_stereo_matching函数
        t1 = time.time()
        result_pic, acc = run_stereo_matching(self.scenario_name, self.matching_algorithm, self.matching_cost,)

        if result_pic is not None:
            # 显示结果
            self.result_display()

            t2 = time.time()
            self.text.append(f"Runtime: {t2 - t1:.3f}s")
            self.text.append(f"The result's size: {str(result_pic.shape)}")
            if acc is not None:
                self.text.append(f"Acc: {acc:.5f}\n")
        else:
            self.text.append("Warning! NCC is not applicable to this picture!\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = MyWindow()
    # display the window
    w.ui.show()

    sys.exit(app.exec_())
