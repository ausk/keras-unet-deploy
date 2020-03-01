// Created by ausk<jincsu#126.com> @ 2020.03.01

#include "unet.h"

int main(int argc, char* argv[]) {
    printf("Usage: xxx unet.xml test.png\n");
    printf("Start test...");
    std::string xmlfpath = "unet.xml";
    std::string imgfpath = "test.png";
    if(argc==3){
        xmlfpath = argv[1];
        imgfpath = argv[2];
    }

    auto infer = UNetInfer(xmlfpath);
    cv::Mat img = cv::imread(imgfpath);
    cv::Mat prob = infer(img);
    cv::imwrite("result.png", prob);

    printf("Great Job!\n");
    return 0;
}