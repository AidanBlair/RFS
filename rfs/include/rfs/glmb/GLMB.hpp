#pragma once
#include "rfs/lmb/LMBComponent.hpp"
#include <Eigen/Dense>
#include <vector>

namespace rfs
{
	using Vector = Eigen::VectorXd;

	struct GLMB
	{
		std::vector<LMBComponent> tt;
		Vector w;
		std::vector<std::vector<int>> I;
		std::vector<int> n;
		Vector cdn;
	};
}