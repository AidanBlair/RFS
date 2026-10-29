#include "rfs/lmb/LMBFilter.hpp"
#include "rfs/models/MotionModel.hpp"
#include "rfs/models/MeasurementModel.hpp"
#include "rfs/models/BirthModel.hpp"
#include "rfs/models/DetectionModel.hpp"
#include "rfs/models/NetworkModel.hpp"
#include "rfs/data/GroundTruth.hpp"
#include "rfs/data/Estimates.hpp"
#include "rfs/data/Measurements.hpp"
#include "rfs/utils/Utils.hpp"
#include "rfs/utils/Metrics.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
	// Initialization
	int number_sensors = 1;
	double communication_range = 500.0;
	rfs::GroundTruth truth = rfs::GroundTruth::from_json("test_ground_truth.json");
	rfs::NetworkModel network_model(number_sensors, communication_range, rfs::NetworkType::Distributed);
	rfs::MotionModel motion_model;
	rfs::MeasurementModel measurement_model(motion_model.x_dim());
	rfs::BirthModel birth_model;
	rfs::DetectionModel detection_model(measurement_model);
	std::vector<rfs::Estimates> ests;
	rfs::Estimates est(truth.K);
	rfs::Measurements meas(truth.K);
	rfs::LMBFilter filter;
	auto pD = detection_model.compute_pD(measurement_model, truth.X[0], false);
	int num_targets = 0;

	// Run time step loop
	for (int k = 0; k < truth.K; ++k)
	{
		std::cout << "\nk = " << k << "\n";
		
		// Adaptive Birth
		if (k > 0)
		{
			birth_model.adaptive_birth(measurement_model, meas, est, k);
		}

		// Prediction Step
		filter.prediction(motion_model, birth_model.birth_components(), k, 0);

		// Prediction Estimation
		filter.set_tt_lmb_predict(glmb2lmb(filter.glmb_predict()));

		// Measurements
		Eigen::MatrixXd measurements = gen_measurements(detection_model, measurement_model, truth.X[k]);
		meas.set(k, measurements);

		// Update
		filter.update(detection_model, measurement_model, motion_model, meas, k, 0);

		// Pruning, Truncation, Track Cleanup
		filter.set_tt_lmb_update(clean_lmb(filter.tt_lmb_update(), filter.track_threshold(), filter.T_max(), filter.number_particles_track()));

		// State Estimation
		rfs::Estimate est_k = extract_estimates(filter.tt_lmb_update(), motion_model.x_dim(), filter.prediction_threshold());
		est.set_X(k, est_k.X);
		est.set_N(k, est_k.N);
		est.set_L(k, est_k.L);

		// Diagnostics
		num_targets = 0;
		for (int tar = 0; tar < filter.tt_lmb_update().size(); ++tar)
		{
			if (filter.tt_lmb_update()[tar].r >= filter.prediction_threshold())
			{
				num_targets++;
			}
		}

		if (num_targets != truth.N[k])
		{
			std::cout << "Cardinality error: truth = " << truth.N[k] << ", estimate = " << num_targets << "\n";
		}
	}
	// Metrics
	const double ospa_c = 50.0;	// OSPA cutoff (m)
	const double ospa_p = 1.0;	// OSPA order
	const int ospa_2_w = 10;	// OSPA(2) window length
	ests.push_back(est);
	std::vector<Eigen::MatrixXd> OSPA_dist = rfs::OSPA(number_sensors, truth.K, truth.X, ests, ospa_c, ospa_p);
	std::cout << "Mean OSPA dist: " << OSPA_dist.at(0).row(0).mean() << "\n";
	std::cout << "Mean OSPA loc: " << OSPA_dist.at(0).row(1).mean() << "\n";
	std::cout << "Mean OSPA card: " << OSPA_dist.at(0).row(2).mean() << "\n";

	std::vector<Eigen::MatrixXd> X_track = rfs::extract_tracks(truth.X, truth.track_list, truth.total_tracks, truth.K);
	std::vector<std::vector<Eigen::MatrixXd>> Y_track(number_sensors);
	for (int n = 0; n < number_sensors; ++n)
	{
		const int labelcount = rfs::countestlabels(ests.at(n), truth.K);
		ests.at(n).set_total_tracks(labelcount);
		rfs::assigncolorarray(ests.at(n), truth, labelcount);
		Y_track.at(n) = rfs::extract_tracks(ests.at(n).X(), ests.at(n).track_list(), ests.at(n).total_tracks(), truth.K);
	}

	std::vector<Eigen::MatrixXd> OSPA2_dist = rfs::OSPA2(number_sensors, truth.K, X_track, Y_track, ospa_c, ospa_p, ospa_2_w);
	std::cout << "Mean OSPA(2) dist: " << OSPA2_dist.at(0).row(0).mean() << "\n";
	std::cout << "Mean OSPA(2) loc: " << OSPA2_dist.at(0).row(1).mean() << "\n";
	std::cout << "Mean OSPA(2) card: " << OSPA2_dist.at(0).row(2).mean() << "\n";

	// X Add Sensor State(s) - locations and orientations
	// X Create est
	// X Create filter parameters
	// X Creat lmb update
	// X Create lmb birth
	// X Create lmb survive
	// X Create lmb prediction
	// X Add all of those to the LMBFilter object (or better way?)
	// X std::vector<LMBComponent> setter methods
	// X Meas object
	// X Probability of detection
	// X Adaptive birth
	// X Prediction
	// X Measurement
	// X Update
	// X Metrics
	// Fix clutter generation to be in FoV, set maximum detection range to range_c second row?
	// EXTENSIONS
	// Visualization - matplotlibcpp?
	// NetworkModel
	// Fusion
	// Control
	return 0;
}