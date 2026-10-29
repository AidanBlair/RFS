#pragma once
#include "rfs/data/Estimates.hpp"

namespace rfs
{
	using Matrix = Eigen::MatrixXd;

	std::vector<Matrix> OSPA(const int N, const int K, const std::vector<Matrix>& X, const std::vector<Estimates>& Y, const double c, const double p);
	std::vector<Matrix> OSPA2(const int N, const int K, const std::vector<Matrix>& X, const std::vector<std::vector<Matrix>>& Y, const double c, const double p, const int w);
}