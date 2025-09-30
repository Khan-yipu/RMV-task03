#include <ceres/ceres.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

// 弹道模型
struct TrajectoryResidual {
    TrajectoryResidual(double x0, double y0, double x, double y, double t) 
        : x0_(x0), y0_(y0), x_(x), y_(y), t_(t) {}

    template <typename T>
    bool operator()(const T* const parameters, T* residual) const {
        // parameters[0] = vx0
        // parameters[1] = vy0
        // parameters[2] = g
        // parameters[3] = k
        
        const T& vx0 = parameters[0];
        const T& vy0 = parameters[1];
        const T& g = parameters[2];
        const T& k = parameters[3];
        
        // 模型公式
        T exp_kt = exp(-k * t_);
        T x_model = T(x0_) + (vx0 / k) * (T(1.0) - exp_kt);
        T y_model = T(y0_) + ((vy0 + g / k) / k) * (T(1.0) - exp_kt) - (g / k) * t_;
        
        // 残差计算
        residual[0] = T(x_) - x_model;
        residual[1] = T(y_) - y_model;
        
        return true;
    }

private:
    const double x0_;
    const double y0_;
    const double x_;
    const double y_;
    const double t_;
};

// 读取轨迹数据
struct TrajectoryPoint {
    double x, y, timestamp;
};

bool ReadTrajectoryData(const std::string& filename, std::vector<TrajectoryPoint>& points) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }
    
    std::string line;
    // 跳过标题行
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // 解析CSV行
        size_t pos1 = line.find(',');
        size_t pos2 = line.find(',', pos1 + 1);
        size_t pos3 = line.find(',', pos2 + 1);
        
        if (pos1 == std::string::npos || pos2 == std::string::npos || pos3 == std::string::npos) {
            continue;
        }
        
        TrajectoryPoint point;
        point.x = std::stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
        point.y = std::stod(line.substr(pos2 + 1, pos3 - pos2 - 1));
        point.timestamp = std::stod(line.substr(pos3 + 1));
        
        points.push_back(point);
    }
    
    file.close();
    return true;
}

int main() {
    // 读取轨迹数据
    std::vector<TrajectoryPoint> points;
    if (!ReadTrajectoryData("../trajectory.csv", points)) {
        std::cerr << "Failed to read trajectory data" << std::endl;
        return -1;
    }
    
    std::cout << "Read " << points.size() << " trajectory points" << std::endl;
    
    // 获取初始位置
    double x0 = points[0].x;  // 166.505
    double y0 = points[0].y;  // 124.416
    
    // 初始速度估计（基于前几帧）
    double dt = points[1].timestamp - points[0].timestamp;
    double vx0 = (points[1].x - points[0].x) / dt;
    double vy0 = (points[1].y - points[0].y) / dt;
    
    // 初始g和k值
    double g = 500.0;  // px/s^2 (在100-1000范围内)
    double k = 0.1;    // 1/s (在0.01-1范围内)
    
    // 参数数组: [vx0, vy0, g, k]
    double parameters[4] = {vx0, vy0, g, k};
    
    std::cout << "Initial parameters:" << std::endl;
    std::cout << "x0 = " << x0 << " px (fixed)" << std::endl;
    std::cout << "y0 = " << y0 << " px (fixed)" << std::endl;
    std::cout << "vx0 = " << vx0 << " px/s" << std::endl;
    std::cout << "vy0 = " << vy0 << " px/s" << std::endl;
    std::cout << "g = " << g << " px/s^2" << std::endl;
    std::cout << "k = " << k << " 1/s" << std::endl;
    
    // 创建Ceres问题
    ceres::Problem problem;
    
    // 添加残差块
    for (const auto& point : points) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<TrajectoryResidual, 2, 4>(
                new TrajectoryResidual(x0, y0, point.x, point.y, point.timestamp));
        problem.AddResidualBlock(cost_function, nullptr, parameters);
    }
    
    // 设置参数边界约束
    // g: 100-1000
    problem.SetParameterLowerBound(parameters, 2, 100.0);
    problem.SetParameterUpperBound(parameters, 2, 1000.0);
    
    // k: 0.01-1.0
    problem.SetParameterLowerBound(parameters, 3, 0.01);
    problem.SetParameterUpperBound(parameters, 3, 1.0);
    
    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-12;
    options.gradient_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;
    
    // 求解
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    std::cout << summary.FullReport() << std::endl;
    
    // 输出结果
    std::cout << "\nFitted parameters:" << std::endl;
    std::cout << "x0 = " << x0 << " px (fixed)" << std::endl;
    std::cout << "y0 = " << y0 << " px (fixed)" << std::endl;
    std::cout << "vx0 = " << parameters[0] << " px/s" << std::endl;
    std::cout << "vy0 = " << parameters[1] << " px/s" << std::endl;
    std::cout << "g = " << parameters[2] << " px/s^2" << std::endl;
    std::cout << "k = " << parameters[3] << " 1/s" << std::endl;
    
    return 0;
}
