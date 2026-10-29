#pragma once
#include "rfs/lmb/LMBComponent.hpp"
#include "rfs/models/MotionModel.hpp"
#include "rfs/models/BirthModel.hpp"
#include "rfs/models/MeasurementModel.hpp"
#include "rfs/models/DetectionModel.hpp"
#include "rfs/utils/Utils.hpp"
#include "rfs/glmb/GLMB.hpp"
#include "rfs/data/Measurements.hpp"
#include <Eigen/Dense>

namespace rfs
{
	using Vector = Eigen::VectorXd;
	using Matrix = Eigen::MatrixXd;
	using RowVector = Eigen::RowVectorXd;

	struct LMBPredictionResult
	{
		std::vector<LMBComponent> birth;
		std::vector<LMBComponent> survive;
	};

	enum class NoiseType
	{
		Noise,
		Noiseless
	};

	class LMBFilter
	{
	public:
		LMBFilter();

		// Getters
		int T_max() const;
		double track_threshold() const;
		int H_birth() const;
		int H_survive() const;
		int H_update() const;
		double prediction_threshold() const;
		int number_particles_track() const;
		double merging_threshold() const;
		const std::vector<LMBComponent>& tt_lmb_update() const;
		const std::vector<LMBComponent>& tt_lmb_birth() const;
		const std::vector<LMBComponent>& tt_lmb_survive() const;
		const std::vector<LMBComponent>& tt_lmb_predict() const;
		const GLMB& glmb_predict() const;

		// Setters
		const void set_tt_lmb_update(const std::vector<LMBComponent>& update);
		const void set_tt_lmb_birth(const std::vector<LMBComponent>& birth);
		const void set_tt_lmb_survive(const std::vector<LMBComponent>& survive);
		const void set_tt_lmb_predict(const std::vector<LMBComponent>& predict);
		const void set_glmb_predict(const GLMB& predict);

		// Main methods
		void prediction(const MotionModel& motion_model, const std::vector<BirthComponent>& birth_components, const int k, const int sen);
		void update(const DetectionModel& detection_model, const MeasurementModel& measurement_model, const MotionModel& motion_model, const Measurements& meas, const int k, const int sen);

		// Helper methods
		const LMBPredictionResult lmb_predict(const std::vector<LMBComponent>& tt_lmb_update, const MotionModel& motion_model, const std::vector<BirthComponent>& birth_components, const int k, const int sen);
		const Matrix gen_mvs(const Vector& m, const Matrix& P, const int num_par) const;
		const Vector compute_pS(const Matrix& x) const;
		const Matrix gen_newstate_fn(const MotionModel& model, const Matrix& x, const NoiseType noise_type) const;
		const GLMB glmb_update(const DetectionModel& detection_model, const MeasurementModel& measurement_model, const Measurements& meas, const int k, const int sen) const;
		const Vector compute_likelihood(const MeasurementModel& measurement_model, const Vector& measurement, const Matrix& X) const;

	private:
		// Filter parameters
		int T_max_;								// Maximum number of tracks
		double track_threshold_;				// Threshold to prune tracks
		int H_birth_;							// Requested number of birth components/hypotheses (for LMB to GLMB casting before update)
		int H_survive_;							// Requested number of surviving components/hypotheses (for LMB to GLMB casting before update)
		int H_update_;							// Requested number of updated components/hypotheses (for GLMB update)
		//int H_max_;								// Cap on number of posterior components/hypotheses
		//double hypothesis_threshold_;			// Pruning threshold for components/hypotheses
		double prediction_threshold_;			// Pruning threshold for estimates
		int number_particles_track_;			// Number of particles per track
		//int number_particles_track_threshold_;	// Threshold on effective number of particles before resampling
		double merging_threshold_;				// Merging threshold
		//double correction_threshold_;			// Correction threshold

		// LMB/GLMB components
		std::vector<LMBComponent> tt_lmb_update_;
		std::vector<LMBComponent> tt_lmb_birth_;
		std::vector<LMBComponent> tt_lmb_survive_;
		std::vector<LMBComponent> tt_lmb_predict_;
		GLMB glmb_predict_;
	};
}