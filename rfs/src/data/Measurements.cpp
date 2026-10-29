#include "rfs/data/Measurements.hpp"

namespace rfs
{
	Measurements::Measurements(int K)
		: Z_(K)  // size K, each Matrix default-constructed (0x0)
		, K_(K)
	{
	}

	void Measurements::set(int k, const Matrix& Zk)
	{
		assert(k >= 0 && k < K_);
		Z_[k] = Zk;
	}

	const Matrix& Measurements::at(int k) const
	{
		assert(k >= 0 && k < K_);
		return Z_[k];
	}
}