#pragma once
#include <Eigen/Dense>

namespace rfs {
	using Vector = Eigen::VectorXd;
	using Matrix = Eigen::MatrixXd;

	struct LMBComponent
	{
		double r;			// Probability of existence [0.0, 1.0]
		Matrix x;			// Particle states (x_dim,number_particles_track_)
		Vector w;			// Particle weights (1,number_particles_track_)
		Eigen::Vector3i l;	// Label [k, s, i]
	};
}