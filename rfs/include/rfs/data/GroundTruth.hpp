#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

namespace rfs
{
	using Matrix = Eigen::MatrixXd;

	struct GroundTruth
	{
		int K;										// Number of time steps
		std::vector<Matrix> X;						// Target states at each time step
		std::vector<int> N;							// Number of targets at each time step
		std::vector<std::vector<int>> L;			// Labels of targets at each time step
		std::vector<std::vector<int>> track_list;	// Tracks at each time step
		int total_tracks;							// Number of tracks

		static GroundTruth from_json(const std::string& filename);	// Load in ground truth data from a json file
	};
}
