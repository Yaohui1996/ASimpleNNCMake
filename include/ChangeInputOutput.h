#ifndef CHANGEINPUTOUTPUT_H
#define CHANGEINPUTOUTPUT_H

#include<vector>
#include<./eigen-3.3.8/Eigen/Dense>

auto changeOutputFormat(const std::vector<int>& labels)->std::vector<Eigen::VectorXd> ;
auto changeInputFormat(const std::vector<std::vector<int>>& images)->std::vector<Eigen::VectorXd> ;

#endif 
