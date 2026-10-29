#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cassert>

namespace rfs
{
	using Vector = Eigen::VectorXd;
	using Matrix = Eigen::MatrixXd;

	class Measurements
	{
	public:
		explicit Measurements(int K);

		void set(int k, const Matrix& Zk);

		const Matrix& at(int k) const;

	private:
		std::vector<Matrix> Z_;	// Measurements at each time step
		int K_;					// Number of time steps
	};
}