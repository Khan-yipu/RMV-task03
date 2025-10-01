#include <ceres/ceres.h>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

// 弹道模型函数
struct TrajectoryModel {
  static double x(double t, double x0, double vx0, double k) {
    return x0 + (vx0 / k) * (1.0 - exp(-k * t));
  }

  static double y(double t, double y0, double vy0, double g, double k) {
    return y0 + ((vy0 + g / k) / k) * (1.0 - exp(-k * t)) - (g / k) * t;
  }
};

// Ceres拟合函数
struct TrajectoryResidual {
  TrajectoryResidual(double tx, double ty, double tt)
      : x_(tx), y_(ty), t_(tt) {}

  template <typename T>
  bool operator()(const T *const parameters, T *residual) const {
    T x0 = parameters[0];
    T y0 = parameters[1];
    T vx0 = parameters[2];
    T vy0 = parameters[3];
    T g = parameters[4];
    T k = parameters[5];

    T predicted_x = x0 + (vx0 / k) * (T(1.0) - exp(-k * t_));
    T predicted_y =
        y0 + ((vy0 + g / k) / k) * (T(1.0) - exp(-k * t_)) - (g / k) * t_;

    // 计算残差
    residual[0] = predicted_x - T(x_);
    residual[1] = predicted_y - T(y_);

    return true;
  }

  static ceres::CostFunction *Create(const double x, const double y,
                                     const double t) {
    return (new ceres::AutoDiffCostFunction<TrajectoryResidual, 2, 6>(
        new TrajectoryResidual(x, y, t)));
  }

  double x_;
  double y_;
  double t_;
};

int main() {
  // 获取图像序列
  std::cout << "Reading image sequence..." << std::endl;
  std::vector<cv::Mat> frames;
  std::vector<std::string> filenames;

  // 读取图像序列
  for (int i = 1; i <= 144; ++i) {
    char filename[256];
    sprintf(filename, "../frames/frame_%04d.png", i);
    cv::Mat frame = cv::imread(filename);
    if (frame.empty()) {
      // 尝试另一种路径
      sprintf(filename, "frames/frame_%04d.png", i);
      frame = cv::imread(filename);
      if (frame.empty()) {
        std::cerr << "Error reading frame " << i << std::endl;
        continue;
      }
    }
    frames.push_back(frame);
    filenames.push_back(filename);
  }

  if (frames.empty()) {
    std::cerr << "No frames loaded" << std::endl;
    return -1;
  }

  std::cout << "Loaded " << frames.size() << " frames" << std::endl;

  double fps = 60.0;
  int total_frames = frames.size();
  std::cout << "FPS: " << fps << ", Total frames: " << total_frames
            << std::endl;

  // 存储轨迹点
  std::vector<cv::Point2f> trajectory_points;
  std::vector<double> timestamps;

  // 处理图像帧，提取蓝色弹丸位置
  for (int frame_count = 0; frame_count < frames.size(); ++frame_count) {
    cv::Mat frame = frames[frame_count];
    double timestamp = frame_count / fps;

    // 转换到HSV颜色空间以便检测蓝色
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // 定义蓝色范围（调整参数以更好地检测弹丸）
    cv::Scalar lower_blue(90, 100, 100);
    cv::Scalar upper_blue(130, 255, 255);

    // 创建掩码
    cv::Mat mask;
    cv::inRange(hsv, lower_blue, upper_blue, mask);

    // 形态学操作去除噪声
    cv::Mat kernel =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // 找到最大的轮廓（假设是弹丸）
    if (!contours.empty()) {
      double max_area = 0;
      int max_contour_idx = -1;

      for (int i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
          max_area = area;
          max_contour_idx = i;
        }
      }

      if (max_contour_idx >= 0 && max_area > 10) { // 面积阈值过滤噪声
        cv::Moments M = cv::moments(contours[max_contour_idx]);
        if (M.m00 != 0) {
          cv::Point2f center(M.m10 / M.m00, M.m01 / M.m00);
          trajectory_points.push_back(center);
          timestamps.push_back(timestamp);

          // 在原图上绘制检测到的点
          cv::circle(frame, center, 5, cv::Scalar(0, 0, 255), -1);
        }
      }
    }

    // 保存标注后的图像（可选）
    char output_filename[256];
    sprintf(output_filename, "frames/annotated_frame_%04d.png",
            frame_count + 1);
    // cv::imwrite(output_filename, frame);

    // 显示处理后的帧（可选）
    // cv::imshow("Frame", frame);
    // cv::imshow("Mask", mask);
    // if (cv::waitKey(1) == 27) break;  // ESC键退出
  }
  cv::destroyAllWindows();

  if (trajectory_points.size() < 10) {
    std::cerr << "Not enough trajectory points detected: "
              << trajectory_points.size() << std::endl;
    return -1;
  }

  std::cout << "Detected " << trajectory_points.size() << " trajectory points"
            << std::endl;

  // 计算初始速度的估计值
  double initial_vx = 0.0;
  double initial_vy = 0.0;
  if (trajectory_points.size() > 10) {
    // 使用前10个点来估计初始速度
    double dt = 1.0 / fps;
    for (int i = 1; i <= 10 && i < trajectory_points.size(); ++i) {
      double vx = (trajectory_points[i].x - trajectory_points[i - 1].x) / dt;
      double vy = (trajectory_points[i].y - trajectory_points[i - 1].y) / dt;
      initial_vx += vx;
      initial_vy += vy;
    }
    initial_vx /= 10;
    initial_vy /= 10;
  } else {
    initial_vx = 250.0;
    initial_vy = -350.0;
  }

  // 基于轨迹点计算g和k的初始估计值
  double initial_g = -300.0; // 更合理的默认估计值（负值表示向下的重力加速度）
  double initial_k = 0.05;   // 更合理的默认估计值

  // 如果有足够的轨迹点，基于轨迹形状计算更合理的g和k估计值
  if (trajectory_points.size() > 20) {
    // 计算轨迹的最高点
    float max_y = trajectory_points[0].y;
    int max_y_index = 0;
    for (int i = 1; i < trajectory_points.size(); ++i) {
      if (trajectory_points[i].y > max_y) {
        max_y = trajectory_points[i].y;
        max_y_index = i;
      }
    }

    // 分析上升段和下降段的数据来估计g和k
    if (max_y_index > 5 && max_y_index < trajectory_points.size() - 5) {
      double dt = 1.0 / fps;

      // 改进的g估计方法：基于整个下降段的加速度
      double falling_dvy = 0;
      int falling_count = 0;

      // 使用更多的下降段点来估计g
      for (int i = max_y_index + 1;
           i <= max_y_index + 15 && i < trajectory_points.size() - 1; ++i) {
        double vy1 = (trajectory_points[i].y - trajectory_points[i - 1].y) / dt;
        double vy2 = (trajectory_points[i + 1].y - trajectory_points[i].y) / dt;
        double acc = (vy2 - vy1) / dt;
        falling_dvy += acc;
        falling_count++;
      }

      // 如果有足够的数据点，使用计算的值
      if (falling_count > 0) {
        double avg_falling_acc = falling_dvy / falling_count;

        // 估计g值（应该接近下降段的加速度的绝对值）
        initial_g = fabs(avg_falling_acc);

        // 确保g在合理范围内
        if (initial_g < 100)
          initial_g = 100;
        if (initial_g > 1000)
          initial_g = 1000;
      }

      // 改进的k估计方法：基于多个时间段的速度衰减
      double k_sum = 0;
      int k_count = 0;

      // 计算多个时间段内的速度衰减
      for (int start_idx = 5; start_idx < trajectory_points.size() - 10;
           start_idx += 5) {
        double v_start = sqrt(pow((trajectory_points[start_idx + 1].x -
                                   trajectory_points[start_idx].x) /
                                      dt,
                                  2) +
                              pow((trajectory_points[start_idx + 1].y -
                                   trajectory_points[start_idx].y) /
                                      dt,
                                  2));
        double v_end = sqrt(pow((trajectory_points[start_idx + 6].x -
                                 trajectory_points[start_idx + 5].x) /
                                    dt,
                                2) +
                            pow((trajectory_points[start_idx + 6].y -
                                 trajectory_points[start_idx + 5].y) /
                                    dt,
                                2));

        if (v_start > 1.0 && v_end > 1.0 &&
            v_end < v_start) { // 确保速度确实衰减了
          double time_interval = 5 * dt;
          // v_end = v_start * exp(-k * time_interval)
          // k = -ln(v_end/v_start) / time_interval
          double k_est = -log(v_end / v_start) / time_interval;
          if (k_est > 0.001 && k_est < 0.5) {
            k_sum += k_est;
            k_count++;
          }
        }
      }

      // 如果有有效的k估计值，使用平均值
      if (k_count > 0) {
        initial_k = k_sum / k_count;
      }

      // 确保k在合理范围内
      if (initial_k < 0.01)
        initial_k = 0.01;
      if (initial_k > 1.0)
        initial_k = 0.06;
    }
  }

  std::cout << "Improved initial estimates: g = " << initial_g
            << ", k = " << initial_k << std::endl;

  std::cout << "Estimated initial velocities: vx0 = " << initial_vx
            << ", vy0 = " << initial_vy << std::endl;
  std::cout << "Estimated initial g = " << initial_g << ", k = " << initial_k
            << std::endl;

  // 初始化参数 [x0, y0, vx0, vy0, g, k]
  double parameters[6] = {
      trajectory_points[0].x, // x0
      trajectory_points[0].y, // y0
      initial_vx,             // vx0 (初始猜测)
      initial_vy,             // vy0 (初始猜测，负值表示向上)
      initial_g,              // g (初始猜测)
      initial_k               // k (初始猜测)
  };

  // 创建Ceres问题
  ceres::Problem problem;

  // 添加残差块
  for (size_t i = 0; i < trajectory_points.size(); ++i) {
    ceres::CostFunction *cost_function = TrajectoryResidual::Create(
        trajectory_points[i].x, trajectory_points[i].y, timestamps[i]);
    problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5),
                             parameters);
  }

  // 设置参数范围约束
  problem.SetParameterLowerBound(parameters, 4, -1000.0); // g >= -1000
  problem.SetParameterUpperBound(parameters, 4, -100.0);  // g <= -100
  problem.SetParameterLowerBound(parameters, 5, 0.01);    // k >= 0.01
  problem.SetParameterUpperBound(parameters, 5, 1.0);     // k <= 1.0

  // 配置求解器
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  // 求解
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // 输出拟合参数
  std::cout << summary.BriefReport() << std::endl;

  std::cout << "Fitted parameters:" << std::endl;
  std::cout << "x0: " << parameters[0] << std::endl;
  std::cout << "y0: " << parameters[1] << std::endl;
  std::cout << "vx0: " << parameters[2] << " px/s" << std::endl;
  std::cout << "vy0: " << parameters[3] << " px/s" << std::endl;
  std::cout << "g: " << parameters[4] << " px/s^2" << std::endl;
  std::cout << "k: " << parameters[5] << " 1/s" << std::endl;

  // 生成拟合轨迹用于绘图
  std::vector<cv::Point2f> fitted_trajectory;
  double dt = 1.0 / fps;
  double max_t = timestamps.back();

  for (double t = 0; t <= max_t; t += dt) {
    double x =
        TrajectoryModel::x(t, parameters[0], parameters[2], parameters[5]);
    double y = TrajectoryModel::y(t, parameters[1], parameters[3],
                                  parameters[4], parameters[5]);
    // 将y坐标取负值以确保上凸形状
    fitted_trajectory.push_back(cv::Point2f(x, -y));
  }

  // 创建对比图
  // 获取轨迹点的边界
  float min_x = trajectory_points[0].x, max_x = trajectory_points[0].x;
  float min_y = -trajectory_points[0].y,
        max_y = -trajectory_points[0].y; // y坐标取负值

  for (const auto &point : trajectory_points) {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, -point.y); // y坐标取负值
    max_y = std::max(max_y, -point.y); // y坐标取负值
  }

  for (const auto &point : fitted_trajectory) {
    min_x = std::min(min_x, point.x);
    max_x = std::max(max_x, point.x);
    min_y = std::min(min_y, point.y);
    max_y = std::max(max_y, point.y);
  }

  // 添加边距
  float margin = 20.0;
  min_x -= margin;
  max_x += margin;
  min_y -= margin;
  max_y += margin;

  // 创建画布 (注意OpenCV的y轴是向下为正，所以需要翻转y坐标)
  int canvas_width = 800;
  int canvas_height = 600;
  cv::Mat canvas(canvas_height, canvas_width, CV_8UC3,
                 cv::Scalar(255, 255, 255));

  // 计算缩放因子和偏移量
  float scale_x = canvas_width / (max_x - min_x);
  float scale_y = canvas_height / (max_y - min_y);
  float scale = std::min(scale_x, scale_y) * 0.9; // 留一些边距

  // 计算偏移量，确保所有点都在画布内
  float offset_x = -min_x + (canvas_width / scale - (max_x - min_x)) / 2;
  float offset_y = -min_y + (canvas_height / scale - (max_y - min_y)) / 2;

  // 绘制检测到的轨迹点
  for (const auto &point : trajectory_points) {
    int x = (int)((point.x + offset_x) * scale);
    int y = canvas_height - (int)((-point.y + offset_y) * scale); // y坐标取负值
    cv::circle(canvas, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1); // 红色点
  }

  // 绘制拟合轨迹
  for (size_t i = 1; i < fitted_trajectory.size(); ++i) {
    int x1 = (int)((fitted_trajectory[i - 1].x + offset_x) * scale);
    int y1 =
        canvas_height - (int)((fitted_trajectory[i - 1].y + offset_y) * scale);
    int x2 = (int)((fitted_trajectory[i].x + offset_x) * scale);
    int y2 = canvas_height - (int)((fitted_trajectory[i].y + offset_y) * scale);
    cv::line(canvas, cv::Point(x1, y1), cv::Point(x2, y2),
             cv::Scalar(255, 0, 0), 2); // 蓝色线
  }

  // 添加图例
  cv::putText(canvas, "Detected points (red)", cv::Point(10, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
  cv::putText(canvas, "Fitted trajectory (blue)", cv::Point(10, 60),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);

  // 保存结果图像
  cv::imwrite("trajectory_comparison.png", canvas);
  std::cout << "Trajectory comparison saved to trajectory_comparison.png"
            << std::endl;

  // 创建单独的轨迹点图像
  cv::Mat points_canvas(canvas_height, canvas_width, CV_8UC3, cv::Scalar(255, 255, 255));

  // 绘制检测到的轨迹点
  for (const auto& point : trajectory_points) {
    int x = (int)((point.x + offset_x) * scale);
    int y = canvas_height - (int)((-point.y + offset_y) * scale);  // y坐标取负值
    cv::circle(points_canvas, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);  // 红色点
  }

  // 添加图例
  cv::putText(points_canvas, "Detected trajectory points", cv::Point(10, 30), 
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

  // 保存轨迹点图像
  cv::imwrite("trajectory_points.png", points_canvas);
  std::cout << "Trajectory points saved to trajectory_points.png" << std::endl;

  // 显示结果图像
  cv::imshow("Trajectory Comparison", canvas);
  cv::waitKey(0);

  return 0;
}
