#ifndef READMNIST_H
#define READMNIST_H

#include <string>
#include <vector>

int ReverseInt(int i);
void read_Mnist_Label(std::string filename, std::vector<int> &labels);
void read_Mnist_Images(std::string filename, std::vector<std::vector<int> > &images);

#endif
