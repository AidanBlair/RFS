#include <rfs/models/BirthModel.hpp>
#include <numbers>

namespace rfs
{
	BirthModel::BirthModel()
	{
		BirthComponent birth_component;
		birth_component.m.resize(5);
		Vector stds(5);

		// 1
		birth_component.L = 1;
		birth_component.r = 0.8;
		birth_component.w = 1;
		birth_component.m << 0.0, 0.0, 295.0, -5.0, 0.0;
		stds << 2.7, 3.8, 2.7, 3.8, 6.0 * std::numbers::pi / 180.0;
		birth_component.B = stds.asDiagonal();
		birth_component.P.resize(5, 5);
		birth_component.P = birth_component.B * birth_component.B.transpose();

		birth_components_.push_back(birth_component);

		// 2
		birth_component.m(0) = 0.0;
		birth_component.m(1) = 0.0;
		birth_component.m(2) = 500.0;
		birth_component.m(3) = 0.0;

		birth_components_.push_back(birth_component);

		// 3
		birth_component.m(0) = 0.0;
		birth_component.m(1) = 0.0;
		birth_component.m(2) = 705.0;
		birth_component.m(3) = 5.0;

		birth_components_.push_back(birth_component);

		// 4
		birth_component.m(0) = -205.0;
		birth_component.m(1) = -5.0;
		birth_component.m(2) = -503.0;
		birth_component.m(3) = -3.0;

		birth_components_.push_back(birth_component);

		// 5
		birth_component.m(0) = -205.0;
		birth_component.m(1) = -5.0;
		birth_component.m(2) = -297.0;
		birth_component.m(3) = 3.0;

		birth_components_.push_back(birth_component);

		// 6
		birth_component.m(0) = -103.0;
		birth_component.m(1) = -3.0;
		birth_component.m(2) = -605.0;
		birth_component.m(3) = -5.0;

		birth_components_.push_back(birth_component);

		// 7
		birth_component.m(0) = -103.0;
		birth_component.m(1) = -3.0;
		birth_component.m(2) = -195.0;
		birth_component.m(3) = 5.0;

		birth_components_.push_back(birth_component);

		// 8
		birth_component.m(0) = 103.0;
		birth_component.m(1) = 3.0;
		birth_component.m(2) = -605.0;
		birth_component.m(3) = -5.0;

		birth_components_.push_back(birth_component);

		// 9
		birth_component.m(0) = 103.0;
		birth_component.m(1) = 3.0;
		birth_component.m(2) = -195.0;
		birth_component.m(3) = 5.0;

		birth_components_.push_back(birth_component);

		// 10
		birth_component.m(0) = 205.0;
		birth_component.m(1) = 5.0;
		birth_component.m(2) = -503.0;
		birth_component.m(3) = -3.0;

		birth_components_.push_back(birth_component);

		// 11
		birth_component.m(0) = 205.0;
		birth_component.m(1) = 5.0;
		birth_component.m(2) = -297.0;
		birth_component.m(3) = 3.0;

		birth_components_.push_back(birth_component);
	}

	const std::vector<BirthComponent>& BirthModel::birth_components() const
	{
		return birth_components_;
	}

	void BirthModel::set_birth_components(std::vector<BirthComponent>& birth_components)
	{
		birth_components_ = birth_components;
	}

	void BirthModel::adaptive_birth(const MeasurementModel& model, const Measurements& meas, const Estimates& est, const int k)
	{
		// Updates model's birth regions at each timestep so that there are birth regions at each measurement and one in the centre
		const Matrix& Zk = meas.at(k);	// 2 x M
		const Matrix& Xprev = (k > 0) ? est.X(k - 1) : Matrix();	// 5 x N

		const auto sensor_pos = model.location();
		const double sensor_range = 1000.0;	// TODO: Change to be in sensor FoV?

		std::vector<BirthComponent> new_birth_components;

		// Centre birth component
		BirthComponent bc;
		bc.L = 1;
		bc.r = 0.002;
		bc.w = 1.0;
		bc.m = Vector(5);
		bc.m << sensor_pos(0), 0.0, sensor_pos(1), 0.0, 0.0;
		Vector stds(5);
		stds << 35.0, 15.0, 35.0, 15.0, 6.0 * std::numbers::pi / 180.0;
		bc.B = stds.asDiagonal();
		bc.P = bc.B * bc.B.transpose();
		new_birth_components.push_back(bc);

		// Loop over measurements
		for (int m = 0; m < Zk.cols(); ++m)
		{
			double x = Zk(0, m);
			double y = Zk(1, m);

			double dist_sensor = std::hypot(sensor_pos(0) - x, sensor_pos(1) - y);

			if (dist_sensor > sensor_range)
			{
				continue;
			}

			int meas_region = 2;

			// Check proximity to existing estimates
			if (Xprev.cols() > 0)
			{
				for (int j = 0; j < Xprev.cols(); ++j)
				{
					double dx = Xprev(0, j) - x;
					double dy = Xprev(2, j) - y;

					if (std::hypot(dx, dy) <= 25.0)
					{
						meas_region = 0;
						break;
					}
				}
			}

			// Create birth component
			BirthComponent bc;
			bc.L = 1;
			bc.r = meas_region * 0.02;
			bc.w = 1.0;
			bc.m = Vector(5);
			bc.m << x, 0.0, y, 0.0, 0.0;
			Vector stds(5);
			stds << 5.0, 5.0, 5.0, 5.0, 10.0 * std::numbers::pi / 180.0;
			bc.B = stds.asDiagonal();
			bc.P = bc.B * bc.B.transpose();
			
			new_birth_components.push_back(bc);
		}

		set_birth_components(new_birth_components);
	}
}
