#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <./eigen-3.3.8/Eigen/Dense>

Eigen::VectorXd sigmoid(const Eigen::VectorXd &a);
Eigen::VectorXd diffSigmoid(const Eigen::VectorXd &z);

#endif
