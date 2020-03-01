// Created  by ausk<jincsu#126.com> @ 2019.11.23
// Modified by ausk<jincsu#126.com> @ 2020.03.01

#if 0
#define WIN32_LEAN_AND_MEAN             // 从 Windows 头文件中排除极少使用的内容
// Windows 头文件
#include <windows.h>

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

#endif 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "unet.h"


// 参考 
// (1) https://pybind11.readthedocs.io/en/master/basics.html
// (2) [191123 使用 Pybind11  和 OpenCV 创建 Python 库](https://zhuanlan.zhihu.com/p/93299698)
namespace py = pybind11;


namespace pybind11 {
namespace detail {


//! (1) 点与元组转换 cv::Point <=> tuple(x,y)
#if 0
template<>
struct type_caster<cv::Point> {
    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));
    bool load(handle obj, bool) {
        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
        return true;
    }

    static handle cast(const cv::Point& pt, return_value_policy, handle) {
        return py::make_tuple(pt.x, pt.y).release();
    }
};

#else
template <>
struct type_caster<cv::Point> {

    //! 定义 cv::Point 类型名为 tuple_xy, 并声明类型为 cv::Point 的局部变量 value。
    PYBIND11_TYPE_CASTER(cv::Point, _("tuple_xy"));

    //! 步骤1：从 Python 转换到 C++。    
    //! 将 Python tuple 对象转换为 C++ cv::Point 类型, 转换失败则返回 false。    
    //! 其中参数2表示是否隐式类型转换。   
    bool load(handle obj, bool) {
        // 确保传参是 tuple 类型        
        if (!py::isinstance<py::tuple>(obj)) {
            std::logic_error("Point(x,y) should be a tuple!");
            return false;
        }

        // 从 handle 提取 tuple 对象，确保其长度是2。        
        py::tuple pt = reinterpret_borrow<py::tuple>(obj);
        if (pt.size() != 2) {
            std::logic_error("Point(x,y) tuple should be size of 2");
            return false;
        }

        //! 将长度为2的 tuple 转换为 cv::Point。        
        value = cv::Point(pt[0].cast<int>(), pt[1].cast<int>());
        return true;
    }

    //! 步骤2： 从 C++ 转换到 Python。    
    //! 将 C++ cv::Mat 对象转换到 tuple，参数2和参数3常忽略。    
    static handle cast(const cv::Point& pt, return_value_policy, handle) {
        return py::make_tuple(pt.x, pt.y).release();
    }
};
#endif 

// (2) 矩形与元组转换 cv::Rect <=> tuple(x,y,w,h)
template<>
struct type_caster<cv::Rect> {

    PYBIND11_TYPE_CASTER(cv::Rect, _("tuple_xywh"));

    bool load(handle obj, bool) {
        if (!py::isinstance<py::tuple>(obj)) {
            std::logic_error("Rect should be a tuple!");
            return false;
        }

        py::tuple rect = reinterpret_borrow<py::tuple>(obj);
        if (rect.size() != 4) {
            std::logic_error("Rect (x,y,w,h) tuple should be size of 4");
            return false;
        }

        value = cv::Rect(rect[0].cast<int>(), rect[1].cast<int>(), rect[2].cast<int>(), rect[3].cast<int>());
        return true;
    }

    static handle cast(const cv::Rect& rect, return_value_policy, handle) {
        return py::make_tuple(rect.x, rect.y, rect.width, rect.height).release();
    }
};

// (3) cv::Mat 与 numpy.ndarray 之间转换
// Python 支持一种通用的插件间数据交换缓冲区协议(buffer protocol)。
// 让类型暴露一个缓冲区视图(buffer view), 这样可以对原始内部数据进行直接访问，常用于矩阵类型。
//
//Pybind11 提供了 py::buffer_info 类型，来映射 Python 缓冲区协议(buffer protocol)。
//
//struct buffer_info {
//    void* ptr;                      /* Pointer to buffer */
//    ssize_t itemsize;               /* Size of one scalar */
//    std::string format;             /* Python struct-style format descriptor */
//    ssize_t ndim;                   /* Number of dimensions */
//    std::vector<ssize_t> shape;     /* Buffer dimensions */
//    std::vector<ssize_t> strides;    /* Strides (in bytes) for each index */
//};

template<>
struct type_caster<cv::Mat> {
public:

    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

    //! 1. cast numpy.ndarray to cv::Mat    
    bool load(handle obj, bool) {
        array b = reinterpret_borrow<array>(obj);
        buffer_info info = b.request();

        int nh = 1;
        int nw = 1;
        int nc = 1;
        int ndims = info.ndim;
        if (ndims == 2) {
            nh = info.shape[0];
            nw = info.shape[1];
        }
        else if (ndims == 3) {
            nh = info.shape[0];
            nw = info.shape[1];
            nc = info.shape[2];
        }
        else {
            throw std::logic_error("Only support 2d, 2d matrix");
            return false;
        }

        int dtype;
        if (info.format == format_descriptor<unsigned char>::format()) {
            dtype = CV_8UC(nc);
        }
        else if (info.format == format_descriptor<int>::format()) {
            dtype = CV_32SC(nc);
        }
        else if (info.format == format_descriptor<float>::format()) {
            dtype = CV_32FC(nc);
        }
        else {
            throw std::logic_error("Unsupported type, only support uchar, int32, float");
            return false;
        }
        value = cv::Mat(nh, nw, dtype, info.ptr);
        return true;
    }

    //! 2. cast cv::Mat to numpy.ndarray    
    static handle cast(const cv::Mat& mat, return_value_policy, handle defval) {
        std::string format = format_descriptor<unsigned char>::format();
        size_t elemsize = sizeof(unsigned char);
        int nw = mat.cols;
        int nh = mat.rows;
        int nc = mat.channels();
        int depth = mat.depth();
        int type = mat.type();
        int dim = (depth == type) ? 2 : 3;
        if (depth == CV_8U) {
            format = format_descriptor<unsigned char>::format();
            elemsize = sizeof(unsigned char);
        }
        else if (depth == CV_32S) {
            format = format_descriptor<int>::format();
            elemsize = sizeof(int);
        }
        else if (depth == CV_32F) {
            format = format_descriptor<float>::format();
            elemsize = sizeof(float);
        }
        else {
            throw std::logic_error("Unsupport type, only support uchar, int32, float");
        }

        std::vector<size_t> bufferdim;
        std::vector<size_t> strides;
        if (dim == 2) {
            bufferdim = { (size_t)nh, (size_t)nw };
            strides = { elemsize * (size_t)nw, elemsize };
        }
        else if (dim == 3) {
            bufferdim = { (size_t)nh, (size_t)nw, (size_t)nc };
            strides = { (size_t)elemsize * nw * nc, (size_t)elemsize * nc, (size_t)elemsize };
        }
        return array(buffer_info(mat.data, elemsize, format, dim, bufferdim, strides)).release();
    }
};

}
}//!


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


PYBIND11_MODULE(pycv, m) {
    m.doc() = "pycv: A small OpenCV binding created using Pybind11"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");
    m.def("sayhello", &sayhello, "Just sayhello!");
    m.def("addpt", &addpt, "add two point");
    m.def("addrect", &addrect, "add two rect");
    m.def("addmat", &addmat, "add two matrix");
    m.def("imread", &imread, "read the file into np.ndarray/cv::Mat");
    m.def("imwrite", &imwrite, "write np.ndarray/cv::Mat into the file");
}