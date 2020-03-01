// Created  by ausk<jincsu#126.com> @ 2019.11.23
// Modified by ausk<jincsu#126.com> @ 2020.03.01

#include "cvbind.h"
#include "unetbind.h"

namespace py = pybind11;

PYBIND11_MODULE(pycv, m) {
    m.doc() = "pycv: A binding created using pybind11";
    bind_CV(m);
    bind_UNet(m);
};
