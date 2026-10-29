#include <rfs/models/NetworkModel.hpp>

namespace rfs
{
	NetworkModel::NetworkModel(int number_sensors, double communication_range, NetworkType network_type)
		: number_sensors_(number_sensors)
		, communication_range_(communication_range)
		, network_type_(network_type)
	{
	}

	int NetworkModel::number_sensors() const
	{
		return number_sensors_;
	}

	double NetworkModel::communication_range() const
	{
		return communication_range_;
	}

	NetworkType NetworkModel::network_type() const
	{
		return network_type_;
	}

	bool NetworkModel::is_centralized() const
	{
		return network_type_ == NetworkType::Centralized;
	}

	bool NetworkModel::is_distributed() const
	{
		return network_type_ == NetworkType::Distributed;
	}
}