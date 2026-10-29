#pragma once
#include "rfs/models/MeasurementModel.hpp"
#include "rfs/data/Measurements.hpp"
#include "rfs/data/Estimates.hpp"
#include <Eigen/Dense>
#include <vector>

namespace rfs
{
	using Vector = Eigen::VectorXd;
	using Matrix = Eigen::MatrixXd;

	struct BirthComponent
	{
		int L;		// Number of Gaussians
		double r;	// Probability of birth
		double w;	// Weight of Gaussians
		Vector m;	// Mean of Gaussians
		Matrix B;	// Standard deviation of Gaussians
		Matrix P;	// Covariance of Gaussians
	};

	class BirthModel {
	public:
		BirthModel();

		// Getter
		const std::vector<BirthComponent>& birth_components() const;

		// Setter
		void set_birth_components(std::vector<BirthComponent>& birth_components);

		// Adaptive Birth
		void adaptive_birth(const MeasurementModel& model, const Measurements& meas, const Estimates& est, const int k);

	private:
		std::vector<BirthComponent> birth_components_;
	};
}
