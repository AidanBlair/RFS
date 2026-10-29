#pragma once
#include <Eigen/Dense>

namespace rfs
{
	class MotionModel
	{
	public:
		using Vector = Eigen::VectorXd;
		using Matrix = Eigen::MatrixXd;

		MotionModel();

		int x_dim() const;
		int v_dim() const;

		int T() const;
		double sigma_vel() const;
		double sigma_turn() const;
		const Matrix& bt() const;
		const Matrix& B2() const;
		const Matrix& B() const;
		const Matrix& Q() const;

	private:
		int x_dim_;  // Dimension of state vector
		int v_dim_;  // Dimension of process noise
		int T_;  // Sampling period
		double sigma_vel_;  // Std of velocity
		double sigma_turn_;  // Std of turn rate variation (rad/s)
		Matrix bt_;
		Matrix B2_;
		Matrix B_;
		Matrix Q_;
	};
}
