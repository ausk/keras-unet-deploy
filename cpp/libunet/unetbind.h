// Created by ausk<jincsu#126.com> @ 2020.03.01

#pragma once
#include "cvbind.h"
#include "unet.h"

namespace py = pybind11;

void bind_UNet(py::module& m);
