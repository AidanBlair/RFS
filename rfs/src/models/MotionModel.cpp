#include <rfs/models/MotionModel.hpp>
#include <numbers>

namespace rfs
{
	MotionModel::MotionModel()
		: x_dim_(5)
		, v_dim_(3)
		, T_(1)
		, sigma_vel_(15.0)
		, sigma_turn_(std::numbers::pi / 180.0)
	{
		bt_.resize(2, 1);
		bt_(0, 0) = (T_ * T_) / 2.0;
		bt_(1, 0) = (double)T_;
		bt_ *= sigma_vel_;

		B2_.resize(x_dim_, v_dim_);
		B2_.setZero();
		B2_.block<2, 1>(0, 0) = bt_;
		B2_.block<2, 1>(2, 1) = bt_;
		B2_(4, 2) = T_ * sigma_turn_;

		B_ = Matrix::Identity(v_dim_, v_dim_);

		Q_ = B_ * B_.transpose();
	}

	int MotionModel::x_dim() const
	{
		return x_dim_;
	}

	int MotionModel::v_dim() const
	{
		return v_dim_;
	}
	
	int MotionModel::T() const
	{
		return T_;
	}

	double MotionModel::sigma_vel() const
	{
		return sigma_vel_;
	}

	double MotionModel::sigma_turn() const
	{
		return sigma_turn_;
	}

	const MotionModel::Matrix& MotionModel::bt() const
	{
		return bt_;
	}

	const MotionModel::Matrix& MotionModel::B2() const
	{
		return B2_;
	}

	const MotionModel::Matrix& MotionModel::B() const
	{
		return B_;
	}

	const MotionModel::Matrix& MotionModel::Q() const
	{
		return Q_;
	}
}
