#include "rfs/lmb/LMBFilter.hpp"
#include "rfs/utils/Utils.hpp"
#include <iostream>
#include <random>
#include <numbers>

namespace rfs
{
	LMBFilter::LMBFilter()
		: T_max_(100)
		, track_threshold_(0.5)
		, H_birth_(4)
		, H_survive_(50)
		, H_update_(50)
		, prediction_threshold_(0.5)
		, number_particles_track_(1000)
		, merging_threshold_(0.1)
	{
	}

	// Getters
	int LMBFilter::T_max() const
	{
		return T_max_;
	}

	double LMBFilter::track_threshold() const
	{
		return track_threshold_;
	}

	int LMBFilter::H_birth() const
	{
		return H_birth_;
	}

	int LMBFilter::H_survive() const
	{
		return H_survive_;
	}

	int LMBFilter::H_update() const
	{
		return H_update_;
	}

	double LMBFilter::prediction_threshold() const
	{
		return prediction_threshold_;
	}

	int LMBFilter::number_particles_track() const
	{
		return number_particles_track_;
	}

	double LMBFilter::merging_threshold() const
	{
		return merging_threshold_;
	}

	const std::vector<LMBComponent>& LMBFilter::tt_lmb_update() const
	{
		return tt_lmb_update_;
	}

	const std::vector<LMBComponent>& LMBFilter::tt_lmb_birth() const
	{
		return tt_lmb_birth_;
	}

	const std::vector<LMBComponent>& LMBFilter::tt_lmb_survive() const
	{
		return tt_lmb_survive_;
	}

	const std::vector<LMBComponent>& LMBFilter::tt_lmb_predict() const
	{
		return tt_lmb_predict_;
	}

	const GLMB& LMBFilter::glmb_predict() const
	{
		return glmb_predict_;
	}

	// Setters
	const void LMBFilter::set_tt_lmb_update(const std::vector<LMBComponent>& update)
	{
		tt_lmb_update_ = update;
	}

	const void LMBFilter::set_tt_lmb_birth(const std::vector<LMBComponent>& birth)
	{
		tt_lmb_birth_ = birth;
	}

	const void LMBFilter::set_tt_lmb_survive(const std::vector<LMBComponent>& survive)
	{
		tt_lmb_survive_ = survive;
	}

	const void LMBFilter::set_tt_lmb_predict(const std::vector<LMBComponent>& predict)
	{
		tt_lmb_predict_ = predict;
	}

	const void LMBFilter::set_glmb_predict(const GLMB& predict)
	{
		glmb_predict_ = predict;
	}

	// Main methods
	void LMBFilter::prediction(const MotionModel& motion_model, const std::vector<BirthComponent>& birth_components, const int k, const int sen)
	{
		// Prediction step for LMB filter
		LMBPredictionResult result = lmb_predict(tt_lmb_update(), motion_model, birth_components, k, sen);
		set_tt_lmb_birth(result.birth);
		set_tt_lmb_survive(result.survive);
		int T_predict = tt_lmb_birth().size() + tt_lmb_survive().size();
		GLMB glmb_predict = castlmbpred(tt_lmb_birth(), tt_lmb_survive(), H_birth(), H_survive());
		set_glmb_predict(glmb_predict);
	}

	void LMBFilter::update(const DetectionModel& detection_model, const MeasurementModel& measurement_model, const MotionModel& motion_model, const Measurements& meas, const int k, const int sen)
	{
		GLMB glmb_update_ = glmb_update(detection_model, measurement_model, meas, k, sen);
		set_tt_lmb_update(glmb2lmb(glmb_update_));
		int T_posterior = tt_lmb_update().size();
	}

	// Helper methods
	const LMBPredictionResult LMBFilter::lmb_predict(const std::vector<LMBComponent>& tt_lmb_update, const MotionModel& motion_model, const std::vector<BirthComponent>& birth_components, const int k, const int sen)
	{
		// LMB prediction of birth and surviving tracks
		int num_par = number_particles_track();

		// Generate birth tracks
		std::vector<LMBComponent> tt_lmb_birth;
		for (size_t i = 0; i < birth_components.size(); ++i)
		{
			LMBComponent LMBC;
			LMBC.x.resize(birth_components[i].m.size(), num_par);
			LMBC.r = birth_components[i].r;
			LMBC.x = gen_mvs(birth_components[i].m, birth_components[i].P, num_par);
			LMBC.w = Eigen::VectorXd::Ones(num_par, 1).array() / num_par;
			LMBC.l = Eigen::Vector3i(k, sen, i);
			tt_lmb_birth.push_back(LMBC);
		}

		// Generate surviving tracks
		std::vector<LMBComponent> tt_lmb_survive;
		for (size_t i = 0; i < tt_lmb_update.size(); ++i)
		{
			LMBComponent LMBC;
			Vector wtemp_predict = compute_pS(tt_lmb_update[i].x).cwiseProduct(tt_lmb_update[i].w);
			Matrix xtemp_predict = gen_newstate_fn(motion_model, tt_lmb_update[i].x, NoiseType::Noise);
			LMBC.r = wtemp_predict.sum() * tt_lmb_update[i].r;
			LMBC.x = xtemp_predict;
			LMBC.w = wtemp_predict / wtemp_predict.sum();
			LMBC.l = tt_lmb_update[i].l;
			tt_lmb_survive.push_back(LMBC);
		}

		LMBPredictionResult result;
		result.birth = std::move(tt_lmb_birth);	// std::move so that the vectors are returned by value, which is good
		result.survive = std::move(tt_lmb_survive);
		return result;
	}

	const Matrix LMBFilter::gen_mvs(const Vector& m, const Matrix& P, const int num_par) const
	{
		// Generates random birth particles
		const int x_dim = m.size();
		Matrix X(x_dim, num_par);

		// Cholesky decomposition: P = U^T * U
		Eigen::LLT<Matrix> llt(P);
		Matrix U = llt.matrixL();

		// Random number generator
		std::normal_distribution<double> dist(0.0, 1.0);

		// Generate samples
		for (int i = 0; i < num_par; ++i)
		{
			Vector z(x_dim);
			for (int d = 0; d < x_dim; ++d)
			{
				z(d) = dist(global_rng());
			}

			X.col(i) = m + U * z;
		}

		return X;
	}

	const Vector LMBFilter::compute_pS(const Matrix& x) const
	{
		// Computes the probability of survival
		Vector pS = Eigen::VectorXd::Constant(x.cols(), 0.99);
		return pS;
	}

	const Matrix LMBFilter::gen_newstate_fn(const MotionModel& model, const Matrix& x, const NoiseType noise_type) const
	{
		// Nonlinear state space equation (CT model)
		if (x.size() == 0)
		{
			return Matrix();
		}

		const int L = x.cols();
		const int T = model.T();
		const double tol = 1e-10;

		// Generate noise V
		Matrix V;
		if (noise_type == NoiseType::Noise)
		{
			std::normal_distribution<double> normal(0.0, 1.0);

			V.resize(model.B().cols(), L);
			for (int i = 0; i < V.rows(); ++i)
			{
				for (int j = 0; j < V.cols(); ++j)
				{
					V(i, j) = normal(global_rng());
				}
			}
		}
		else if (noise_type == NoiseType::Noiseless)
		{
			V = Eigen::MatrixXd::Zero(model.B().cols(), L);
		}

		// Pre-compute quantities
		RowVector omega = x.row(4);
		
		RowVector sin_omega_T = (omega.array() * T).sin();
		RowVector cos_omega_T = (omega.array() * T).cos();

		RowVector a = RowVector::Constant(L, T);
		RowVector b = RowVector::Zero(L);

		for (int i = 0; i < L; ++i)
		{
			if (std::abs(omega(i)) > tol)
			{
				a(i) = sin_omega_T(i) / omega(i);
				b(i) = (1.0 - cos_omega_T(i)) / omega(i);
			}
		}

		// State propagation
		Matrix X(5, L);
		X.setZero();

		// x/vx
		X.row(0) = (x.row(0).array() + a.array() * x.row(1).array() - b.array() * x.row(3).array()).matrix();

		X.row(1) = (cos_omega_T.array() * x.row(1).array() - sin_omega_T.array() * x.row(3).array()).matrix();

		// y/vy
		X.row(2) = (x.row(2).array() + b.array() * x.row(1).array() + a.array() * x.row(3).array()).matrix();

		X.row(3) = (sin_omega_T.array() * x.row(1).array() + cos_omega_T.array() * x.row(3).array()).matrix();

		// Turn rate
		X.row(4) = x.row(4);

		// Add scaled noise
		X += model.B2() * V;

		return X;
	}

	const GLMB LMBFilter::glmb_update(const DetectionModel& detection_model, const MeasurementModel& measurement_model, const Measurements& meas, const int k, const int sen) const
	{
		// Create updated tracks (single target Bayes update)
		GLMB glmb_update;

		const int m = meas.at(k).cols();	// Number of measurements
		const int number_predicted_tracks = glmb_predict().tt.size();

		std::vector<LMBComponent> tt_update((1 + m) * number_predicted_tracks);
		for (int tabidx = 0; tabidx < number_predicted_tracks; ++tabidx)
		{
			tt_update[tabidx] = glmb_predict().tt[tabidx];
		}

		Matrix allcostm = Eigen::MatrixXd::Zero(number_predicted_tracks, m);

		for (int emm = 0; emm < m; ++emm)
		{
			for (int tabidx = 0; tabidx < number_predicted_tracks; ++tabidx)
			{
				int stoidx = number_predicted_tracks * (emm + 1) + tabidx;	// index of predicted track i updated with measurement j is (number_predicted_tracks * j + i)

				Vector w_temp = detection_model.compute_pD(measurement_model, glmb_predict().tt[tabidx].x, false).array() * glmb_predict().tt[tabidx].w.array() * compute_likelihood(measurement_model, meas.at(k).col(emm), glmb_predict().tt[tabidx].x).array();

				Matrix x_temp = glmb_predict().tt[tabidx].x;
				tt_update[stoidx].x = x_temp;						// Particles for updated track
				tt_update[stoidx].w = w_temp.array() / w_temp.sum();			// Weights of particles for updated track
				tt_update[stoidx].l = glmb_predict().tt[tabidx].l;	// Track label
				allcostm(tabidx, emm) = w_temp.sum();				// Predictive likelihood
			}
		}

		glmb_update.tt = tt_update;	// Copy tack table back to GLMB struct

		// Precalculation loop for average detection/missed probabilities
		Vector avpd = Eigen::VectorXd::Zero(number_predicted_tracks);

		for (int tabidx = 0; tabidx < number_predicted_tracks; ++tabidx)
		{
			avpd(tabidx) = glmb_predict().tt[tabidx].w.dot(detection_model.compute_pD(measurement_model, glmb_predict().tt[tabidx].x, false)) + std::numeric_limits<double>::denorm_min();
		}

		Vector avqd = Eigen::VectorXd::Ones(avpd.size()) - avpd;

		// Component updates
		if (m == 0)	// No measurements means all missed detections
		{
			for (int pidx = 0; pidx < glmb_predict().w.size(); ++pidx)
			{
				Vector hypotheses(glmb_predict().I[pidx].size());
				for (size_t i = 0; i < glmb_predict().I[pidx].size(); ++i)
				{
					hypotheses(i) = avqd(glmb_predict().I[pidx][i]);
				}
				glmb_update.w.conservativeResize(pidx + 1);
				glmb_update.w(pidx) = -measurement_model.clutter_intensity() + hypotheses.array().log().sum() + std::log(glmb_predict().w[pidx]);	// hypothesis/component weight
			}
			glmb_update.I = glmb_predict().I;	// hypothesis/component tracks (via indices to track table)
			glmb_update.n = glmb_predict().n;	// hypothesis/component cardinality
		}
		else
		{
			// loop over predicted hypotheses/components
			int runidx = 0;
			for (int pidx = 0; pidx < glmb_predict().w.size(); ++pidx)
			{
				if (glmb_predict().n[pidx] == 0)	// No target means all clutter
				{
					glmb_update.w.conservativeResize(runidx + 1);
					glmb_update.w(runidx) = -measurement_model.clutter_intensity() + m * std::log(measurement_model.clutter_intensity() * measurement_model.clutter_pdf()) + std::log(glmb_predict().w[pidx]);
					glmb_update.I.push_back(glmb_predict().I[pidx]);
					glmb_update.n.push_back(glmb_predict().n[pidx]);
					runidx++;
				}
				else
				{
					// Otherwise perform update for component
					// Calculate best updated hypotheses/components
					Matrix temp_1(glmb_predict().I[pidx].size(), allcostm.cols());
					for (int i = 0; i < glmb_predict().I[pidx].size(); ++i)
					{
						temp_1.row(i) = allcostm.row(glmb_predict().I[pidx][i]);
					}
					Vector temp_2(glmb_predict().I[pidx].size());
					for (int i = 0; i < glmb_predict().I[pidx].size(); ++i)
					{
						temp_2(i) = avqd(glmb_predict().I[pidx][i]);
					}
					Matrix costm = temp_1.array() / (measurement_model.clutter_intensity() * measurement_model.clutter_pdf() * temp_2.replicate(1, m)).array();	// Cost matrix
					Matrix neglogcost = (-1 * (costm.array().log())).matrix();	// Negative log cost
					double threshold = 100.0;
					for (int i = 0; i < neglogcost.rows(); ++i) {
						for (int j = 0; j < neglogcost.cols(); ++j) {
							if (neglogcost(i, j) > threshold) {
								neglogcost(i, j) = std::numeric_limits<double>::infinity();
							}
						}
					}

					int num_best = std::round(H_update() * std::sqrt(glmb_predict().w(pidx)) / glmb_predict().w.array().sqrt().sum());

					MurtyResult mr = mbestwrap_updt_custom(eigenToStdVector(neglogcost), num_best);	// Murty's algorithm to calculate m-best assignment hypotheses/components

					// Generate corresponding surviving hypotheses/components
					for (int hidx = 0; hidx < mr.costs.size(); ++hidx)
					{
						std::vector<int> update_hypcmp_tmp = mr.assignments[hidx];
						for (int& val : update_hypcmp_tmp)
						{
							val++;	// NECESSARY TO PREVENT ERRORS
							val *= glmb_predict().tt.size();
						}
						std::vector<int> result(update_hypcmp_tmp.size());
						for (size_t i = 0; i < update_hypcmp_tmp.size(); ++i)
						{
							result[i] = update_hypcmp_tmp[i] + glmb_predict().I[pidx][i];
						}
						glmb_update.w.conservativeResize(runidx + 1);
						glmb_update.w(runidx) = -measurement_model.clutter_intensity() + m * std::log(measurement_model.clutter_intensity() * measurement_model.clutter_pdf()) + temp_2.array().log().sum() + std::log(glmb_predict().w[pidx]) - mr.costs[hidx];	// Hypothesis/component weight
						glmb_update.I.push_back(result);	// Hypothesis/component tracks (via indices to track table)
						glmb_update.n.push_back(glmb_predict().n[pidx]);	// Hypothesis/component cardinality
						runidx++;
					}
				}
			}
		}

		glmb_update.w = (glmb_update.w.array() - std::log(glmb_update.w.array().exp().sum())).exp();	// Normalize weights

		// Extract cardinality distribution
		int max_card = *std::max_element(glmb_update.n.begin(), glmb_update.n.end());
		for (int card = 0; card < max_card; ++card)
		{
			double sum_w = 0.0;
			for (size_t i = 0; i < glmb_update.n.size(); ++i) {
				if (glmb_update.n[i] == card) {
					sum_w += glmb_update.w(i);
				}
			}
			glmb_update.cdn.resize(card + 1);
			glmb_update.cdn(card) = sum_w;	// Extract probability of n targets
		}

		return glmb_update;
	}

	const Vector LMBFilter::compute_likelihood(const MeasurementModel& measurement_model, const Vector& measurement, const Matrix& X) const
	{
		// Compute likelihood vector
		const int M = X.cols();
		Matrix P = Eigen::MatrixXd::Zero(2, X.cols());
		P.row(0) = X.row(0);
		P.row(1) = X.row(2);
		
		Vector inv_std = measurement_model.noise_std().diagonal().cwiseInverse();	// Inverse of diagonal matrix noise_std (D)
		//Vector inv_var =
		//	measurement_model.noise_std()
		//	.diagonal()
		//	.array()
		//	.square()
		//	.inverse();

		Matrix diff = measurement.replicate(1, M) - P;	// Difference between particles and measurement

		Vector e_sq = (diff.array().colwise() * inv_std.array()).square().colwise().sum().transpose();	// Compute squared Mahalanobis distance (Vector of length M)
		//Vector e_sq =
		//	(diff.array().colwise() * inv_var.array())
		//	.square()
		//	.colwise()
		//	.sum()
		//	.transpose();

		double norm_const = std::log(2.0 * std::numbers::pi * measurement_model.noise_std().diagonal().prod());	// Normalization constant

		Vector gz_vals = (-0.5 * e_sq.array() - norm_const).exp();	// Final likelihood values

		return gz_vals;
	}
}