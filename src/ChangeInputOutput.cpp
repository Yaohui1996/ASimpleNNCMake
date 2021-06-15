#include "../include/ChangeInputOutput.h"
#include <vector>
#include <../include/eigen-3.3.8/Eigen/Dense>

using std::vector;

vector<Eigen::VectorXd>changeOutputFormat(const vector<int> &labels) 
{
    vector<Eigen::VectorXd> labels_eigen;
    vector<Eigen::VectorXd>::size_type N = labels.size();
    for (vector<Eigen::VectorXd>::size_type i = 0; i != N; ++i)
    {
        Eigen::VectorXd temp_Vector = Eigen::VectorXd::Zero(10); //输出层神经元数目
        Eigen::Index footNum = labels[i];                        //可能会丢失精度
        temp_Vector[footNum] = 1.0;
        labels_eigen.emplace_back(temp_Vector);
    }
    return labels_eigen;
}

vector<Eigen::VectorXd> changeInputFormat(const vector<vector<int>> &images) 
{
    vector<Eigen::VectorXd> images_eigen;
    for (vector<vector<int>>::const_iterator it = images.cbegin(); it != images.cend(); ++it)
    {
        Eigen::VectorXd temp = Eigen::VectorXd::Zero((*it).size());
        for (Eigen::VectorXd::Index j = 0; j != (*it).size(); ++j)
        {
            temp(j) = (*it)[j];
        }
        temp /= 255.0; //归一化
        images_eigen.emplace_back(temp);
    }
    return images_eigen;
}
