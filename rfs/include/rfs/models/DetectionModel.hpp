#pragma once
#include <Eigen/Dense>
#include "MeasurementModel.hpp"

namespace rfs
{
	using Array = Eigen::ArrayXd;
	using Vector = Eigen::VectorXd;
	using Matrix = Eigen::MatrixXd;

	class DetectionModel
	{
	public:
		explicit DetectionModel(const MeasurementModel& model);

		// Getters
		double max_detection_probability() const;
		double min_rho() const;
		double max_rho() const;
		double max_theta() const;
		double k_rho() const;
		double k_theta() const;

		// Functions
		Vector sigmoid_pD(const Vector& x, const double x_max, const double k_x, const double y_max) const;
		Vector compute_pD(const MeasurementModel& model, const Matrix& X, const bool pims) const;

	private:
		double max_detection_probability_;	// Max probability of detection [0, 1]
		double min_rho_;					// Minimum detection range (usually 0)
		double max_rho_;					// Maximum detection range
		double max_theta_;					// Maximum angle (from heading angle, symmetric) for detection
		double k_rho_;						// Sigmoid parameter for range
		double k_theta_;					// Sigmoid parameter for angle
	};
}