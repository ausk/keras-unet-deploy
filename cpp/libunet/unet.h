// Created by ausk<jincsu#126.com> @ 2020.03.01

#pragma once
#include "ai.h"

class UNetInfer {
	IAI* punet;
	std::string model_xmlfpath;

public:

	UNetInfer(std::string xmlfpath){
		model_xmlfpath = xmlfpath;
		punet = nullptr;
		init_UNet();
	}

	~UNetInfer() {
		destroy_UNet();
	}

	cv::Mat operator()(const cv::Mat& src) {
		return run_UNet(src);
	}

	void init_UNet();
	void destroy_UNet();
	cv::Mat run_UNet(const cv::Mat&);
};
