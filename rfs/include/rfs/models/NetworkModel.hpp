#pragma once

namespace rfs
{
	enum class NetworkType
	{
		Centralized,
		Distributed
	};

	class NetworkModel
	{
	public:
		explicit NetworkModel(int number_sensors, double communication_range, NetworkType network_type);

		int number_sensors() const;
		double communication_range() const;
		NetworkType network_type() const;
		bool is_centralized() const;
		bool is_distributed() const;

	private:
		int number_sensors_;
		double communication_range_;
		NetworkType network_type_;
	};
}