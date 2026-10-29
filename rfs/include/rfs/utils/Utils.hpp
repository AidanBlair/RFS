#pragma once
#include "rfs/glmb/GLMB.hpp"
#include "rfs/lmb/LMBComponent.hpp"
#include "rfs/lmb/LMBFilter.hpp"
#include "rfs/data/Estimates.hpp"
#include "rfs/data/GroundTruth.hpp"
#include "rfs/data/Measurements.hpp"
#include "rfs/models/DetectionModel.hpp"
#include "rfs/utils/Hungarian.hpp"
#include <random>

namespace rfs
{
	using Vector = Eigen::VectorXd;
	using Matrix = Eigen::MatrixXd;
	using LabelMatrix = Eigen::Matrix<int, Eigen::Dynamic, 3>;

	struct KShortestPathResult
	{
		std::vector<std::vector<int>> shortest_paths;
		std::vector<double> total_costs;
	};

	struct PathCost
	{
		std::vector<int> path;	// shortest path from source to destination
		double cost;			// total path cost
	};

	struct CandidatePath
	{
		int path_index;			// index in P
		std::vector<int> path;	// candidate path
		double cost;			// total cost
	};

	struct UniqueRowsResult
	{
		LabelMatrix cu;			// unique rows
		std::vector<int> ic;	// index mapping
	};

	struct UniqueRowsResultXd
	{
		Matrix cu;				// unique rows
		std::vector<int> ic;	// index mapping
	};

	// Hashable row type
	struct Row3
	{
		int a, b, c;

		bool operator==(const Row3& other) const
		{
			return a == other.a && b == other.b && c == other.c;
		}
	};

	// Hash function
	struct Row3Hash
	{
		std::size_t operator()(const Row3& r) const noexcept
		{
			std::size_t h1 = std::hash<int>{}(r.a);
			std::size_t h2 = std::hash<int>{}(r.b);
			std::size_t h3 = std::hash<int>{}(r.c);

			// Good hash combine
			return h1 ^ (h2 << 1) ^ (h3 << 2);
		}
	};

	// RGB
	struct RGB
	{
		double r, g, b;
	};

	// Color Array
	struct ColorArray
	{
		std::unordered_map<Row3, int, Row3Hash> label_to_idx;
		std::vector<RGB> rgb;
	};

	// Estimate
	struct Estimate
	{
		Matrix X;
		int N;
		Eigen::MatrixXi L;
	};

	// Murty Result
	struct MurtyResult
	{
		std::vector<std::vector<int>> assignments;	// m x nRows
		std::vector<double> costs;					// size m
	};

	//const std::vector<std::vector<int>>& generate_graph(std::vector<MeasurementModel> models, const double communication_range) const;
	// LMB to GLMB
	const GLMB castlmbpred(const std::vector<LMBComponent>& tt_lmb_birth, const std::vector<LMBComponent>& tt_lmb_survive, const int H_bth, const int H_sur);
	const GLMB lmb2glmb(const std::vector<LMBComponent>& tt_lmb, const int H_req);
	const Vector get_rvals(const std::vector<LMBComponent>& tt_lmb);
	const KShortestPathResult kshortestpath_pred(const Vector& rs, const int k);
	const KShortestPathResult kshortestpath_any(const Matrix& net_cost_matrix, const int source, const int destination, const int k_paths);
	const PathCost BFMSpath(const Matrix& G, const int source, const int destination);

	// GLMB to LMB
	const std::vector<LMBComponent> glmb2lmb(const GLMB& glmb);
	const UniqueRowsResult unique_rows_stable_label(const LabelMatrix& lmat);
	const UniqueRowsResultXd unique_rows_stable_numeric(const Matrix& X);
	const Vector accumarray(const std::vector<int>& ic, const Vector& w);

	// Estimates
	const Estimate extract_estimates(const std::vector<LMBComponent>& tt_lmb, const int x_dim, const double prediction_threshold);

	// Measurements
	const Matrix gen_measurements(const DetectionModel& detection_model, const MeasurementModel& model, const Matrix& truth_X);
	const Matrix gen_observation(const MeasurementModel& measurement_model, const Matrix& truth_X);
	const Matrix keep_observations_in_fov(const DetectionModel& detection_model, const MeasurementModel& measurement_model, const Matrix& observations);
	const Matrix convert_observations_to_states(const MeasurementModel& measurement_model, const Matrix& observations);
	const Matrix gen_clutter(const DetectionModel& detection_model, const MeasurementModel& measurement_model);

	// Random
	std::mt19937& global_rng();
	Vector randn(const int N);
	Vector rand(const int N, const int a, const int b);

	// Assignment
	MurtyResult mbestwrap_updt_custom(const std::vector<std::vector<double>>& P0, int m);
	MurtyResult murty_custom(const std::vector<std::vector<double>>& P0, int m);
	std::pair<std::vector<int>, double> solve_assignment(const std::vector<std::vector<double>>& P);
	std::vector<std::vector<double>> eigenToStdVector(const Matrix& mat);

	// Pruning
	const std::vector<LMBComponent> clean_lmb(const std::vector<LMBComponent>& tt_lmb_in, const double track_threshold, const int T_max, const int number_particles_track);
	std::vector<int> systematic_resample(const Eigen::VectorXd& w, const int N);

	// Metric helpers
	std::vector<Matrix> extract_tracks(const std::vector<Matrix>& X, const std::vector<std::vector<int>>& track_list, const int total_tracks, const int K);
	int countestlabels(Estimates& est, const int K);
	ColorArray makecolorarray(int labelcount);
	void assigncolorarray(Estimates& est, const GroundTruth& truth, int labelcount);
}