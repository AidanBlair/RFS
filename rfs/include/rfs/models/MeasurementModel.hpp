#pragma once
#include <Eigen/Dense>

namespace rfs
{
	class MeasurementModel {
	public:
		using Vector = Eigen::VectorXd;
		using Matrix = Eigen::MatrixXd;

		explicit MeasurementModel(int x_dim);  // TODO later: Load values from a file rather than hardcoded

		// Dimensions
		int x_dim() const;
		int z_dim() const;
		int w_dim() const;

		// Measurement function
		const Matrix& noise_std() const;
		const Matrix& noise_covariance() const;
		const Matrix& coordinate_extractor() const;

		// Detection & clutter
		double detection_probability() const;
		double misdetection_probability() const;
		double clutter_intensity() const;
		const Eigen::Matrix<double, 2, 2>& clutter_range() const;
		double clutter_pdf() const;

		// Sensor geometry
		const Eigen::Vector2d& location() const;
		double range() const;
		double heading() const;
		double fov() const;

		void set_location(const Eigen::Vector2d& location);
		void set_heading(double angle);

	private:
		int x_dim_;  // Dimension of state vector
		int z_dim_;  // Dimension of observation vector
		int w_dim_;  // Dimension of observation noise

		Matrix D_;  // Std for x and y noise
		Matrix R_;  // Covariance for observation noise
		Matrix C_;  // Extracts coordinates

		double P_D_;  // Probability of detection in measurements
		double lambda_c_;  // Poisson average rate of uniform clutter (per scan)
		Eigen::Matrix<double, 2, 2> range_c_;  // Uniform clutter range for x and y
		double pdf_c_;  // Uniform clutter density

		Eigen::Vector2d location_;  // Sensor location (2x1 vector)
		double range_;  // Sensor range (m)
		double heading_;  // Sensor heading [0, 2*pi]
		double fov_;  // Sensor FOV [0, 2*pi]
	};
}
