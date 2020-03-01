// Created by ausk<jincsu#126.com> @ 2020.03.01

#include "ai.h"

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

namespace IE = InferenceEngine;

// 将 BGR 通道的图像(uchar)拷贝到 IE::Blob (float) 中
void mat2blob(cv::Mat& mat, IE::Blob::Ptr& blob, double alpha /* = 1.0*/, double beta/* = 0.0*/) {
	//! 拷贝到输入中
	//IE::Blob::Ptr fblob = infer_request.GetBlob(input_name);
	if (mat.channels() == 1 && mat.isContinuous()) {
		cv::Mat xinput;
		mat.convertTo(xinput, CV_32F, alpha, beta);
		std::copy_n((float*)xinput.data, xinput.cols * xinput.rows, (float*)blob->buffer().as<float*>());
	}
	else {
		float* pdst = (float*)blob->buffer().as<float*>();
		//float* psrc = (float*)mat.data; //mat.ptr();
		uchar* psrc = (uchar*)mat.data; //mat.ptr();
		int nheight = mat.rows;
		int nwidth = mat.cols;
		int nchannel = mat.channels();

		for (int iy = 0; iy < nheight; ++iy) {
			for (int ix = 0; ix < nwidth; ++ix) {
				for (int ic = 0; ic < nchannel; ++ic) {
					pdst[ic * nheight * nwidth + iy * nwidth + ix] = beta + alpha * psrc[iy * nwidth * nchannel + ix * nchannel + ic];
				}
			}
		}
	}
}
