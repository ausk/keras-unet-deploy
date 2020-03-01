// Created by ausk<jincsu#126.com> @ 2020.03.01

#include "unetbind.h"

void bind_UNet(py::module& m){
    py::class_<UNetInfer>(m, "UNetInfer")
        .def(py::init<const std::string&>())
        .def("__call__", &UNetInfer::operator())
        .def("init_UNet", &UNetInfer::init_UNet)
        .def("destroy_UNet", &UNetInfer::destroy_UNet);
}