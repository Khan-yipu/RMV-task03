#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

// 结构体存储轨迹点
struct TrajectoryPoint {
    int frame;
    cv::Point2d position;
    double timestamp;
};

// 检测弹丸位置
cv::Point2d detectProjectile(const cv::Mat& frame, const cv::Mat& prevFrame) {
    // 使用背景差分检测运动物体
    cv::Mat diff;
    cv::absdiff(frame, prevFrame, diff);
    
    // 转换为灰度图
    cv::Mat gray;
    if (diff.channels() == 3) {
        cv::cvtColor(diff, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = diff.clone();
    }
    
    // 二值化
    cv::Mat binary;
    cv::threshold(gray, binary, 30, 255, cv::THRESH_BINARY);
    
    // 形态学操作去除噪声
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 找到最大的轮廓（假设弹丸是最大的运动物体）
    if (!contours.empty()) {
        auto max_contour = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        
        // 计算轮廓的中心
        cv::Moments m = cv::moments(*max_contour);
        if (m.m00 != 0) {
            return cv::Point2d(m.m10 / m.m00, m.m01 / m.m00);
        }
    }
    
    // 如果没有检测到，返回(-1, -1)
    return cv::Point2d(-1, -1);
}

int main() {
    // 设置帧目录
    std::string frames_dir = "../frames";
    
    // 收集所有帧文件
    std::vector<std::string> frame_files;
    for (const auto& entry : fs::directory_iterator(frames_dir)) {
        if (entry.path().extension() == ".png") {
            frame_files.push_back(entry.path().string());
        }
    }
    
    // 按文件名排序
    std::sort(frame_files.begin(), frame_files.end());
    
    if (frame_files.empty()) {
        std::cerr << "No frame files found in " << frames_dir << std::endl;
        return -1;
    }
    
    std::cout << "Found " << frame_files.size() << " frames" << std::endl;
    
    // 读取第一帧作为参考
    cv::Mat prev_frame = cv::imread(frame_files[0]);
    if (prev_frame.empty()) {
        std::cerr << "Failed to read first frame: " << frame_files[0] << std::endl;
        return -1;
    }
    
    // 存储轨迹点
    std::vector<TrajectoryPoint> trajectory;
    
    // FPS为60
    const double fps = 60.0;
    const double dt = 1.0 / fps;
    
    // 处理每一帧
    for (size_t i = 1; i < frame_files.size(); ++i) {
        cv::Mat frame = cv::imread(frame_files[i]);
        if (frame.empty()) {
            std::cerr << "Failed to read frame: " << frame_files[i] << std::endl;
            continue;
        }
        
        // 检测弹丸位置
        cv::Point2d position = detectProjectile(frame, prev_frame);
        
        // 如果检测到有效位置，记录轨迹点
        if (position.x >= 0 && position.y >= 0) {
            TrajectoryPoint point;
            point.frame = i;
            point.position = position;
            point.timestamp = i * dt;
            trajectory.push_back(point);
        }
        
        // 更新前一帧
        prev_frame = frame.clone();
        
        // 显示进度
        if (i % 20 == 0) {
            std::cout << "Processed " << i << "/" << frame_files.size() << " frames" << std::endl;
        }
    }
    
    std::cout << "Detected " << trajectory.size() << " trajectory points" << std::endl;
    
    // 保存轨迹到CSV文件
    std::ofstream csv_file("trajectory.csv");
    if (!csv_file.is_open()) {
        std::cerr << "Failed to create trajectory.csv" << std::endl;
        return -1;
    }
    
    // 写入CSV头部
    csv_file << "frame,x,y,timestamp" << std::endl;
    
    // 写入轨迹数据
    for (const auto& point : trajectory) {
        csv_file << point.frame << ","
                << point.position.x << ","
                << point.position.y << ","
                << point.timestamp << std::endl;
    }
    
    csv_file.close();
    
    std::cout << "Trajectory saved to trajectory.csv" << std::endl;
    
    return 0;
}