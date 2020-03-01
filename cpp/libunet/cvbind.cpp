// Created  by ausk<jincsu#126.com> @ 2019.11.23
// Modified by ausk<jincsu#126.com> @ 2020.03.01

#include "cvbind.h"

// 基本的加法
int add(int i, int j) {
    return i + j;
}

// 输出 hello
void sayhello() {
    printf("Hello!\n");
}

//! 点相加
cv::Point addpt(cv::Point& lhs, cv::Point& rhs) {
    return cv::Point(lhs.x + rhs.x, lhs.y + rhs.y);
}

//! 矩形相加
cv::Rect addrect(cv::Rect& lhs, cv::Rect& rhs) {
    int x = std::min(lhs.x, rhs.x);
    int y = std::min(lhs.y, rhs.y);
    int width = std::max(lhs.x + lhs.width - 1, rhs.x + rhs.width - 1) - x + 1;
    int height = std::max(lhs.y + lhs.height - 1, rhs.y + rhs.height - 1) - y + 1;
    return cv::Rect(x, y, width, height);
}

//! 矩阵相加
cv::Mat addmat(cv::Mat& lhs, cv::Mat& rhs) {
    return lhs + rhs;
}

//！读取图片
cv::Mat imread(std::string fpath) {
    return cv::imread(fpath);
}

//! 写入图片
void imwrite(std::string fpath, const cv::Mat& img) {
    cv::imwrite(fpath, img);
}

// 分离模块
// https://pybind11.readthedocs.io/en/master/faq.html#how-can-i-reduce-the-build-time
void bind_CV(py::module& m) {
    // Python + OpenCV
    m.def("add", &add, "A function which adds two numbers");
    m.def("sayhello", &sayhello, "Just sayhello!");
    m.def("addpt", &addpt, "add two point");
    m.def("addrect", &addrect, "add two rect");
    m.def("addmat", &addmat, "add two matrix");
    m.def("imread", &imread, "read the file into np.ndarray/cv::Mat");
    m.def("imwrite", &imwrite, "write np.ndarray/cv::Mat into the file");
}

/*
PYBIND11_MODULE(pycv, m) {
    m.doc() = "pycv: A small OpenCV binding created using Pybind11"; // optional module docstring
    bind_CV(m);
};
*/