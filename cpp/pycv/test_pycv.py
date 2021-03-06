# 2019/11/23 初始版本
# 2020/03/01 增加 unet 推理功能
# Created  by ausk<jincsu#126.com> @ 2019.11.23
# Modified by ausk<jincsu#126.com> @ 2020.03.01

# (1) https://pybind11.readthedocs.io/en/master/basics.html
# (2) [191123 使用 Pybind11  和 OpenCV 创建 Python 库](https://zhuanlan.zhihu.com/p/93299698)

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(curdir, "../x64/Release"))
import pycv
import numpy as np

# 测试绑定的 OpenCV
def test_cv():
    print("\n===== Test pybind11 for OpenCV =====")
    # 矩阵相加
    a = np.random.random((2,3)).astype(np.float32)
    b = np.ones((2,3)).astype(np.float32)
    c = pycv.addmat(a, b)
    print("a = \n", a)
    print("b = \n", b)
    print("c = \n", c)

    # 点相加
    a = (1,2)
    b = (3,-1)
    c = pycv.addpt(a, b)
    print("{} + {} = {}".format(a, b, c))

    # 矩形相加
    a = (10, 20, 20, 10) # lt(10, 20) => rb(30, 30)
    b = (5, 5, 20, 20)   # lt(5, 5)   => rb(25, 25)
    c = pycv.addrect(a, b) # lt(5, 5)   => rb(30, 30) => wh(25, 25)
    print("a = \n", a)
    print("b = \n", b)
    print("c = \n", c)
    print("===== Finish =====\n")

# 测试使用 pybind11 绑定的 OpenVino+OpenCV 实现 UNet 推理
def test_unet():
    print("\n===== Test pybind11 for UNet (OpenCV+OpenVino) =====")
    xmlfpath = "../unet/unet.xml"
    imgfpath = "../unet/test.png"

    img = pycv.imread(imgfpath)
    unet = pycv.UNetInfer(xmlfpath)
    res = unet(img)
    pycv.imwrite("result.png", res)
    print("Saved into result.png")
    print("===== Finish =====\n")

if __name__ == "__main__":
    test_cv()
    test_unet()
    print("\nDone!")