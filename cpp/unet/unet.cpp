// Created by ausk<jincsu#126.com> @ 2020.03.01

#include "unet.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

namespace IE = InferenceEngine;

class UNet : public IAI {
public:
	IE::Core ie;
	IE::InferRequest infer_request;
	IE::ExecutableNetwork exec_network;
	IE::InputsDataMap input_info;
	IE::OutputsDataMap output_info;
	std::string input_name;
	std::string output_name;
	int channel = 1;
	int height = 256;
	int width = 256;
public:
	virtual bool initialize();
	virtual bool run(const AIDict& src, AIDict& dst);
};

bool UNet::initialize() {
	IE::CNNNetwork network = ie.ReadNetwork(model_xmlfpath, model_binfpath);

	network.setBatchSize(1);       //! 设置批大小为 1

	input_info = network.getInputsInfo();
	output_info = network.getOutputsInfo();

	std::cout << "=> Preparing input/output blobs" << std::endl;
	assert((input_info.size() == 1) && (output_info.size() == 1));
	auto input_item = *input_info.begin();
	auto output_item = *output_info.begin();

	input_item.second->setPrecision(IE::Precision::FP32);
	input_item.second->setLayout(IE::Layout::NCHW);
	output_item.second->setPrecision(IE::Precision::FP32);

	input_name = input_item.first;
	output_name = output_item.first;
	auto input_dims = input_item.second->getTensorDesc().getDims();
	auto output_dims = output_item.second->getTensorDesc().getDims();

	channel = input_dims[1];
	height = input_dims[2];
	width = input_dims[3];

	std::cout << "=> Load network into plugin and create inferRequest" << std::endl;
	//ie.AddExtension(std::make_shared<IE::Extensions::Cpu::CpuExtensions>(), "CPU");

	exec_network = ie.LoadNetwork(network, device);
	infer_request = exec_network.CreateInferRequest();
	std::cout << "[ INFO ] Created UNet infer_request !\n\n";

	return true;
}

bool UNet::run(const AIDict& srcdict, AIDict& dstdict) {
	assert(srcdict.find("image") != srcdict.end());
	cv::Mat src = srcdict.find("image")->second.clone();

	//! 预处理
	assert(!src.empty() && (src.channels() == 1 || src.channels() == 3));
	cv::Mat gray;
	if (src.channels() == 3) {
		cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	}
	else {
		src.copyTo(gray);
	}

	cv::resize(gray, gray, cv::Size(width, height));
	//cv::equalizeHist(gray, gray);

	std::cout << "graysize: " << gray.size() << std::endl;

	//! 拷贝到输入中
	IE::Blob::Ptr input = infer_request.GetBlob(input_name);
	mat2blob(gray, input, 1.0f / 255);

	//! 推理
	int count = 1;
	auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < count; ++i) {
		infer_request.Infer();
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	auto dt = std::chrono::duration<double>(t1 - t0).count();
	std::cout << "[ INFO ] infering cost " << dt * 1000 / count << "ms" << std::endl;


	//! 后处理
	IE::Blob::Ptr output_blob = infer_request.GetBlob(output_name);

	std::vector<std::vector<cv::Point>> cnts;
	std::vector<std::vector<cv::Point>> xcnts;

	//! 先缩放, 后检测
	cv::Mat prob = cv::Mat(cv::Size(width, height), CV_32F, output_blob->buffer().as<float*>());
	prob = prob.clone();
	prob.convertTo(prob, CV_8U, 255.0);
	cv::resize(prob, prob, src.size());

	dstdict.clear();
	dstdict.insert({ "prob", prob.clone() });

	return true;
}



//! ============================================================================
//! 初始化 Unet
void UNetInfer::init_UNet() {
	if (punet) {
		delete punet;
	}
	punet = new UNet;
	punet->setDevice("CPU");
	punet->setModel(model_xmlfpath);
	bool flag = punet->initialize();
	std::cout << "initialized: " << flag << std::endl;
}

// 销毁
void UNetInfer::destroy_UNet() {
	if (punet) {
		delete punet;
	}
}

// 运行推理
cv::Mat UNetInfer::run_UNet(const cv::Mat& img) {
	cv::Mat ximg;
	if (img.channels() == 3) {
		cv::cvtColor(img, ximg, cv::COLOR_BGR2GRAY);
	}
	else {
		img.copyTo(img);
	}

	AIDict srcdict, dstdict;

	std::cout << "imgsize: " << ximg.size() << std::endl;
	srcdict.insert({ "image", ximg });
	punet->run(srcdict, dstdict);

	cv::Mat prob;
	AIDict::iterator piter;
	AIDict::iterator pend = dstdict.end();
	if ((piter = dstdict.find("prob")) != pend) {
		//piter->second.copyTo(dst);
		std::swap(piter->second, prob);
	}
	return prob;
}