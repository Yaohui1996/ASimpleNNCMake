#include "../include/Hadamard.h"
#include <../include/eigen-3.3.8/Eigen/Dense>

Eigen::VectorXd hadamard(const Eigen::VectorXd &v1, const Eigen::VectorXd &v2)
{
    Eigen::VectorXd ret = Eigen::VectorXd::Zero(v1.size());
    for (Eigen::VectorXd::Index i = 0; i != v1.size(); ++i)
    {
        ret[i] = v1[i] * v2[i];
    }
    return ret;
}
