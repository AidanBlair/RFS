#include "rfs/models/DetectionModel.hpp"
#include <cmath>
#include <numbers>
#include <iostream>

namespace rfs
{
	DetectionModel::DetectionModel(const MeasurementModel& model)
		: max_detection_probability_(0.99)
		, min_rho_(0.0)
		, k_rho_(1.0)
		, k_theta_(20.0)
	{
		max_rho_ = model.range();
		max_theta_ = model.fov();
	}

	// Getters
	double DetectionModel::max_detection_probability() const
	{
		return max_detection_probability_;
	}

	double DetectionModel::min_rho() const
	{
		return min_rho_;
	}

	double DetectionModel::max_rho() const
	{
		return max_rho_;
	}

	double DetectionModel::max_theta() const
	{
		return max_theta_;
	}

	double DetectionModel::k_rho() const
	{
		return k_rho_;
	}

	double DetectionModel::k_theta() const
	{
		return k_theta_;
	}

	// Functions
	Vector DetectionModel::sigmoid_pD(const Vector& x, const double x_max, const double k_x, const double y_max) const
	{
		//std::cout << "x: " << x << "\n";
		//std::cout << "x_max: " << x_max << "\n";
		//std::cout << "k_x: " << k_x << "\n";
		//std::cout << "y_max: " << y_max << "\n";
		return y_max * (1.0 / (1.0 + (k_x * (x.array() - x_max)).exp()));
	}

	Vector DetectionModel::compute_pD(const MeasurementModel& model, const Matrix& X, const bool pims) const
	{
		// Initialize
		int n = X.cols();
		Vector pD = Vector::Zero(n, 1);

		// Vectorized range calculation
		Matrix P(2, X.cols());
		P.row(0) = X.row(0);
		P.row(1) = X.row(2);
		Eigen::Vector2d M = model.location();
		Vector rho = (P.colwise() - M).colwise().norm();

		// Vectorized angle calculation
		Vector range_x = P.row(0).array() - M(0);
		Vector range_y = P.row(1).array() - M(1);
		Vector theta(range_x.size());
		for (int i = 0; i < range_x.size(); ++i)
		{
			theta(i) = std::atan2(range_y(i), range_x(i));
		}

		// Pre-compute min/max angles
		double min_angle = model.heading() - model.fov();
		double max_angle = model.heading() + model.fov();

		// Adjust theta for FOV comparison
		Vector theta_adj = theta;
		for (int i = 0; i < theta.size(); ++i)
		{
			if (theta(i) < min_angle)
			{
				theta_adj(i) += 2.0 * std::numbers::pi;
			}
			if (theta(i) > max_angle)
			{
				theta_adj(i) -= 2.0 * std::numbers::pi;
			}
		}

		// Vectorizes heading calculation
		Vector heading = (theta_adj.array() - model.heading()).cwiseAbs();
		for (int i = 0; i < heading.size(); ++i)
		{
			if (heading(i) < std::numbers::pi)
			{
				heading(i) += 2 * std::numbers::pi;
			}
			if (heading(i) > std::numbers::pi)
			{
				heading(i) -= 2 * std::numbers::pi;
			}
			heading(i) = std::abs(heading(i));
		}
		//std::cout << "headings: " << heading << "\n";

		// Vectorized calculations
		//Vector pD = Vector::Zero(rho.size());

		// Create mask for valid indices
		Eigen::Array<bool, Eigen::Dynamic, 1> valid = (theta_adj.array() >= min_angle)
													&& (theta_adj.array() <= max_angle)
													&& (rho.array() >= min_rho())
													&& (rho.array() <= max_rho());

		// Set pD values
		if (pims)
		{
			pD = valid.cast<double>();	// Set 1 for valid, 0 for invalid
		}
		else
		{
			// Compute sigmoid for valid points
			Array rho_vals = sigmoid_pD(rho.array(), max_rho(), k_rho(), max_detection_probability());
			Array theta_vals = sigmoid_pD(heading.array(), max_theta(), k_theta(), max_detection_probability());
			//std::cout << "rho_vals: " << rho_vals << "\n";
			//std::cout << "theta_vals: " << theta_vals << "\n";

			Array rho_sigmoids = valid.select(rho_vals, Array::Zero(valid.size()));
			Array theta_sigmoids = valid.select(theta_vals, Array::Zero(valid.size()));
			pD = (rho_sigmoids * theta_sigmoids).matrix();	// implicitly converts from ArrayXd to VectorXd, but I'm explicitly casting it here
		}

		// Return detection probabilities
		return pD;
	}
}