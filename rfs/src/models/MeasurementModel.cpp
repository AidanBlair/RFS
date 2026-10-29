#include <rfs/models/MeasurementModel.hpp>
#include <numbers>

namespace rfs
{
	MeasurementModel::MeasurementModel(int x_dim)
		: x_dim_(x_dim)
		, z_dim_(2)
		, w_dim_(2)
		, P_D_(0.99)
		, lambda_c_(5.0)
		, range_(5000)
		, heading_(0.0 * std::numbers::pi)
		//, fov_(0.25 * std::numbers::pi)
		, fov_(std::numbers::pi)
	{
		// Measurement matrix (extract x, y)
		C_.resize(z_dim_, x_dim_);
		C_ << 1, 0, 0, 0, 0,
			  0, 0, 1, 0, 0;

		// Measurement noise (std dev)
		D_.resize(z_dim_, z_dim_);
		D_.setZero();
		D_(0, 0) = 2.0;
		D_(1, 1) = 2.0;

		// Noise covariance
		R_ = D_ * D_.transpose();

		// Clutter
		range_c_ << -2000, 2000,
					-2000, 2000;
		pdf_c_ = 1.0 / ((range_c_(0, 1) - range_c_(0, 0)) * (range_c_(1, 1) - range_c_(1, 0)));

		// Geometry
		location_ = Eigen::Vector2d{ 0.0, 0.0 };
		location_ = Eigen::Vector2d{ -250.0, 500.0 };
	}

	// Dimensions

	int MeasurementModel::x_dim() const
	{
		return x_dim_;
	}

	int MeasurementModel::z_dim() const
	{
		return z_dim_;
	}

	int MeasurementModel::w_dim() const
	{
		return w_dim_;
	}

	// Measurement

	const MeasurementModel::Matrix& MeasurementModel::noise_std() const
	{
		return D_;
	}

	const MeasurementModel::Matrix& MeasurementModel::noise_covariance() const
	{
		return R_;
	}

	const MeasurementModel::Matrix& MeasurementModel::coordinate_extractor() const
	{
		return C_;
	}

	// Detection & clutter

	double MeasurementModel::detection_probability() const
	{
		return P_D_;
	}

	double MeasurementModel::misdetection_probability() const
	{
		return 1.0 - P_D_;
	}

	double MeasurementModel::clutter_intensity() const
	{
		return lambda_c_;
	}

	const Eigen::Matrix<double, 2, 2>& MeasurementModel::clutter_range() const
	{
		return range_c_;
	}

	double MeasurementModel::clutter_pdf() const
	{
		return pdf_c_;
	}

	// Geometry

	const Eigen::Vector2d& MeasurementModel::location() const
	{
		return location_;
	}

	double MeasurementModel::range() const
	{
		return range_;
	}

	double MeasurementModel::heading() const
	{
		return heading_;
	}

	double MeasurementModel::fov() const
	{
		return fov_;
	}

	void MeasurementModel::set_location(const Eigen::Vector2d& location)
	{
		location_ = location;
	}

	void MeasurementModel::set_heading(double angle)
	{
		heading_ = angle;
	}
}
