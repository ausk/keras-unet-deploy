// Created by ausk<jincsu#126.com> @ 2020.03.01

#pragma once
#include <string>
#include <algorithm>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

namespace IE = InferenceEngine;

// 获取无后缀文件名
inline std::string fileNameWithoutExt(const std::string& fpath) {
	return fpath.substr(0, std::min<size_t>(fpath.size(), fpath.rfind('.')));
}

// 将 BGR 通道的图像(uchar)拷贝到 IE::Blob (float) 中
void mat2blob(cv::Mat& mat, IE::Blob::Ptr& blob, double alpha = 1.0f, double beta = 0);

//! 定义 AI 接口类
typedef std::unordered_map<std::string, cv::Mat> AIDict;

class IAI {
public:
	virtual bool run(const AIDict& srcdict, AIDict& dstdict) = 0;
	virtual bool setModel(const std::string& model_xmlfpath) {
		this->model_xmlfpath = model_xmlfpath;
		this->model_binfpath = fileNameWithoutExt(model_xmlfpath) + ".bin";
		return true;
	}
	virtual bool setDevice(const std::string& device) {
		this->device = device;
		return true;
	}
	virtual bool initialize() { return false; }
	virtual ~IAI() {};

public:
	std::string model_xmlfpath;
	std::string model_binfpath;
	std::string device;
};