#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "../include/ReadMNIST.h"
#include "../include/ChangeInputOutput.h"
#include "../include/ActivationFunction.h"
#include "../include/Hadamard.h"
#include <../include/eigen-3.3.8/Eigen/Dense>

using std::cout;
using std::endl;
using std::vector;

int main()
{
    //初始化模型
    vector<int> neuresofLayer{784, 64, 128, 64, 10}; //神经网络层数和每层的单元数
    constexpr int iterNums = 30;                     //训练次数
    int L = neuresofLayer.size();                    //神经网络层数
    double eta = 0.1;                                //负梯度下降步长
    constexpr double alpha = 0.9;                    //每一轮训练的步长衰减系数

    //读取数据
    vector<vector<int> > train_images;
    vector<vector<int> > test_images;
    vector<int> train_labels;
    vector<int> test_labels;

    cout << "准备读取训练集图像数据：" << endl;
    read_Mnist_Images("../data/train-images.idx3-ubyte", train_images);
    cout << "训练集图像数据读取完毕！" << endl;

    cout << "准备读取训练集标签数据：" << endl;
    read_Mnist_Label("../data/train-labels.idx1-ubyte", train_labels);
    cout << "训练集标签数据读取完毕！" << endl;

    cout << "准备读取测试集图像数据：" << endl;
    read_Mnist_Images("../data/t10k-images.idx3-ubyte", test_images);
    cout << "测试集图像数据读取完毕！" << endl;

    cout << "准备读取测试集标签数据：" << endl;
    read_Mnist_Label("../data/t10k-labels.idx1-ubyte", test_labels);
    cout << "测试集标签数据读取完毕！" << endl;

    vector<Eigen::VectorXd> trainImageData = changeInputFormat(train_images);
    vector<Eigen::VectorXd> trainLabelData = changeOutputFormat(train_labels);
    vector<Eigen::VectorXd> testImageData = changeInputFormat(test_images);
    vector<Eigen::VectorXd> testLabelData = changeOutputFormat(test_labels);

    //随机数生成
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 g1(seed1);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    auto slcg = [&]()
    { return distribution(g1); };

    //初始化a
    vector<Eigen::VectorXd> a(L);
    for (vector<Eigen::VectorXd>::size_type i = 0; i != a.size(); ++i)
    {
        a[i] = Eigen::VectorXd::Zero(neuresofLayer[i]);
    }
    //初始化w
    vector<Eigen::MatrixXd> w(L - 1);
    for (vector<Eigen::MatrixXd>::size_type i = 0; i != w.size(); ++i)
    {
        w[i] = Eigen::MatrixXd::NullaryExpr(neuresofLayer[i + 1], neuresofLayer[i], slcg);
    }

    //初始化z
    vector<Eigen::VectorXd> z(L - 1);
    for (vector<Eigen::VectorXd>::size_type i = 0; i != z.size(); ++i)
    {
        z[i] = Eigen::VectorXd::Zero(neuresofLayer[i + 1]);
    }

    //初始化b
    vector<Eigen::VectorXd> b(L - 1);
    for (vector<Eigen::VectorXd>::size_type i = 0; i != b.size(); ++i)
    {
        b[i] = Eigen::VectorXd::NullaryExpr(neuresofLayer[i + 1], slcg);
    }
    //初始化delta
    vector<Eigen::VectorXd> delta(L - 1);
    for (vector<Eigen::VectorXd>::size_type i = 0; i != delta.size(); ++i)
    {
        delta[i] = Eigen::VectorXd::Zero(neuresofLayer[i + 1]);
    }
    //初始化partialC_partialb
    vector<Eigen::VectorXd> partialC_partialb(L - 1);
    for (vector<Eigen::VectorXd>::size_type i = 0; i != partialC_partialb.size(); ++i)
    {
        partialC_partialb[i] = Eigen::VectorXd::Zero(neuresofLayer[i + 1]);
    }
    //初始化partialC_partialw
    vector<Eigen::MatrixXd> partialC_partialw(L - 1);
    for (vector<Eigen::MatrixXd>::size_type i = 0; i != partialC_partialw.size(); ++i)
    {
        partialC_partialw[i] = Eigen::MatrixXd::Zero(neuresofLayer[i + 1], neuresofLayer[i]);
    }

    //训练模型
    for (int iter = 0; iter != iterNums; ++iter)
    {
        cout << "当前开始第 " << iter << " 轮训练" << endl;
        for (vector<Eigen::VectorXd>::size_type n = 0; n != trainImageData.size(); ++n)
        {
            if (n % 10000 == 0)
                cout << "正在投喂第 " << n << " 个样本！" << endl;

            //前向传播
            a[0] = trainImageData[n];
            for (vector<int>::size_type i = 0; i != L - 1; ++i)
            {
                z[i] = w[i] * a[i] + b[i];
                a[i + 1] = sigmoid(z[i]);
            }

            //反向计算delta
            for (int i = L - 2; i >= 0; --i)
            {
                i == L - 2 ? delta[i] = hadamard(2 * (a[i + 1] - trainLabelData[n]), diffSigmoid(z[i])) : delta[i] = hadamard(w[i + 1].transpose() * delta[i + 1], diffSigmoid(z[i]));
            }

            //计算梯度方向
            for (vector<Eigen::VectorXd>::size_type i = 0; i != partialC_partialb.size(); ++i)
            {
                partialC_partialb[i] = delta[i];
            }
            for (vector<Eigen::MatrixXd>::size_type i = 0; i != partialC_partialw.size(); ++i)
            {
                partialC_partialw[i] = delta[i] * a[i].transpose();
            }

            //更新坐标
            for (vector<Eigen::VectorXd>::size_type i = 0; i != partialC_partialb.size(); ++i)
            {
                b[i] -= eta * partialC_partialb[i];
            }

            for (vector<Eigen::MatrixXd>::size_type i = 0; i != partialC_partialw.size(); ++i)
            {
                w[i] -= eta * partialC_partialw[i];
            }
        }
        cout << endl;

        eta = alpha * eta;

        //训练集准确度
        int right_counter = 0;
        for (vector<Eigen::VectorXd>::size_type n = 0; n != trainImageData.size(); ++n)
        {
            // if (n % 10000 == 0)
            //    cout << "正在测试训练集中第 " << n << "个样本是否准确！" << endl;
            //前向传播
            a[0] = trainImageData[n];
            for (vector<int>::size_type i = 0; i != L - 1; ++i)
            {
                z[i] = w[i] * a[i] + b[i];
                a[i + 1] = sigmoid(z[i]);
            }
            Eigen::VectorXd::Index max_aL_index;
            a[a.size() - 1].maxCoeff(&max_aL_index);

            Eigen::VectorXd real_val = trainLabelData[n];
            Eigen::VectorXd pre_val = Eigen::VectorXd::Zero(neuresofLayer[L - 1]);
            pre_val[max_aL_index] = 1.0;

            if ((real_val - pre_val).norm() < 0.01)
            {
                right_counter += 1;
            }
        }
        cout << "训练集准确率：" << right_counter / 60000.0 << endl;
        cout << endl;

        //测试集准确度
        int right_counter_test = 0;
        for (vector<Eigen::VectorXd>::size_type n = 0; n != testImageData.size(); ++n)
        {
            // if (n % 10000 == 0)
            //    cout << "正在测试测试集中第 " << n << "个样本是否准确！" << endl;
            //前向传播
            a[0] = testImageData[n];
            for (vector<int>::size_type i = 0; i != L - 1; ++i)
            {
                z[i] = w[i] * a[i] + b[i];
                a[i + 1] = sigmoid(z[i]);
            }
            Eigen::VectorXd::Index max_aL_index;
            a[a.size() - 1].maxCoeff(&max_aL_index);

            Eigen::VectorXd real_val = testLabelData[n];
            Eigen::VectorXd pre_val = Eigen::VectorXd::Zero(neuresofLayer[L - 1]);
            pre_val[max_aL_index] = 1.0;

            if ((real_val - pre_val).norm() < 0.01)
            {
                right_counter_test += 1;
            }
        }
        cout << "测试集准确率：" << right_counter_test / 10000.0 << endl;
        cout << endl;
    }
}
