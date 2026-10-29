#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cassert>

namespace rfs
{
	using Vector = Eigen::VectorXd;
	using Matrix = Eigen::MatrixXd;

	class Estimates
	{
	public:
		explicit Estimates(int K);

		// Setters (per time step)
		void set_X(int k, const Matrix& Xk);
		void set_N(int k, int Nk);
		void set_L(int k, const Eigen::MatrixXi& Lk);
		void set_total_tracks(int labelcount);
		void set_track_list(int k, const std::vector<int>& track);

		// Getters
		const std::vector<Matrix>& X() const;
		const Matrix& X(int k) const;
		int N(int k) const;
		const Eigen::MatrixXi& L(int k) const;
		int total_tracks() const;
		std::vector<std::vector<int>> track_list() const;
		int K() const;

	private:
		std::vector<Matrix> X_;						// Estimated target states at each time step
		std::vector<int> N_;						// Estimated number of targets at each time step
		std::vector<Eigen::MatrixXi> L_;			// Estimate labels of targets at each time step
		int total_tracks_;							// Estimate total number of tracks
		std::vector<std::vector<int>> track_list_;	// Estimate track list
		int K_;										// Number of time steps
	};
}