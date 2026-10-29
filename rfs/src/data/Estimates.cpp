#include "rfs/data/Estimates.hpp"
#include <iostream>

namespace rfs
{
	Estimates::Estimates(int K)
		: X_(K)				// size K, each Matrix default-constructed (0x0)
		, N_(K)				// size K, all zeros
		, L_(K)				// size K, each Matrix default-constructed (0x0)
		, total_tracks_()	// size 1, int
		, track_list_(K)	// size K, each Matrix default-constructed (0x0)
		, K_(K)				// number of time steps
	{
	}

	void Estimates::set_X(int k, const Matrix& Xk)
	{
		assert(k >= 0 && k < K_);
		X_[k] = Xk;
	}

	void Estimates::set_N(int k, int Nk)
	{
		assert(k >= 0 && k < K_);
		N_[k] = Nk;
	}

	void Estimates::set_L(int k, const Eigen::MatrixXi& Lk)
	{
		assert(k >= 0 && k < K_);
		L_[k] = Lk;
	}

	void Estimates::set_total_tracks(int labelcount)
	{
		assert(labelcount >= 0);
		total_tracks_ = labelcount;
	}

	void Estimates::set_track_list(int k, const std::vector<int>& track)
	{
		assert(k >= 0 && k < K_);
		track_list_[k] = track;
	}

	const std::vector<Matrix>& Estimates::X() const
	{
		return X_;
	}

	const Matrix& Estimates::X(int k) const
	{
		assert(k >= 0 && k < K_);
		return X_[k];
	}

	int Estimates::N(int k) const
	{
		assert(k >= 0 && k < K_);
		return N_[k];
	}

	const Eigen::MatrixXi& Estimates::L(int k) const
	{
		assert(k >= 0 && k < K_);
		return L_[k];
	}

	int Estimates::total_tracks() const
	{
		return total_tracks_;
	}

	std::vector<std::vector<int>> Estimates::track_list() const
	{
		return track_list_;
	}

	int Estimates::K() const
	{
		return K_;
	}
}