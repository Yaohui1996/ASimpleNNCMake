#ifndef READMNIST_H
#define READMNIST_H

#include <string>
#include <vector>

auto ReverseInt(int i) -> int;
auto read_Mnist_Label(std::string filename, std::vector<int> &labels) -> void;
auto read_Mnist_Images(std::string filename, std::vector<std::vector<int>> &images) -> void;

#endif 
