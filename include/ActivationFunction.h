#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <./eigen-3.3.8/Eigen/Dense>

auto sigmoid(const Eigen::VectorXd &a) -> Eigen::VectorXd;
auto diffSigmoid(const Eigen::VectorXd &z) -> Eigen::VectorXd;

#endif
