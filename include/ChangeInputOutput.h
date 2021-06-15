#ifndef CHANGEINPUTOUTPUT_H
#define CHANGEINPUTOUTPUT_H

#include<vector>
#include<./eigen-3.3.8/Eigen/Dense>

std::vector<Eigen::VectorXd> changeOutputFormat(const std::vector<int>& labels);
std::vector<Eigen::VectorXd> changeInputFormat(const std::vector<std::vector<int>>& images);

#endif 
