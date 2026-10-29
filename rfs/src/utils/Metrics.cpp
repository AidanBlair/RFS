#include "rfs/utils/Metrics.hpp"
#include "rfs/utils/Utils.hpp"
#include <numeric>

namespace rfs
{
	std::vector<Matrix> OSPA(const int N, const int K, const std::vector<Matrix>& X, const std::vector<Estimates>& Y, const double c, const double p)
	{
		std::vector<Matrix> OSPA_dist(N, Matrix(3, K));	// N = number of sensors, K = number of time steps

		for (int n = 0; n < N; ++n)	// For each sensor
		{
			for (int k = 0; k < K; ++k)	// For each time step
			{
				if ((X.at(k).size() == 0) && (Y.at(n).X(k).size() == 0))	// If there are no targets in both the estimate and the ground truth
				{
					Vector v(3);
					v << 0.0, 0.0, 0.0;
					OSPA_dist.at(n).col(k) = v;
					break;
				}

				if ((X.at(k).size() == 0) || (Y.at(n).X(k).size() == 0))	// If there are no targets in just one of the estimate and the ground truth
				{
					Vector v(3);
					v << c, 0.0, c;
					OSPA_dist.at(n).col(k) = v;
					break;
				}

				int X_size = X.at(k).cols();
				int Y_size = Y.at(n).X(k).cols();

				// Calculate cost matrix for pairings
				Matrix D(X_size, Y_size);

				for (int i = 0; i < X_size; ++i)
				{
					for (int j = 0; j < Y_size; ++j)
					{
						double dist = (X.at(k).col(i) - Y.at(n).X(k).col(j)).norm();
						D(i, j) = std::pow(std::min(c, dist), p);
					}
				}

				// Compute optimal assignment and cost using the Hungarian algorithm
				auto [assignment, cost] = solve_assignment(eigenToStdVector(D));

				// Calculate final distance
				const double cardinality_cost = std::pow(c, p) * std::abs(Y_size - X_size);
				double dist = std::pow(1.0 / std::max(X_size, Y_size) * (cardinality_cost + cost), 1.0 / p);
				double loc = std::pow(1.0 / std::max(X_size, Y_size) * cost, 1.0 / p);
				double card = std::pow(1.0 / std::max(X_size, Y_size) * cardinality_cost, 1.0 / p);
				Vector result(3);
				result << dist, loc, card;
				OSPA_dist.at(n).col(k) = result;
			}
		}

		return OSPA_dist;
	}

	std::vector<Matrix> OSPA2(const int N, const int K, const std::vector<Matrix>& X_track, const std::vector<std::vector<Matrix>>& Y_track, const double c, const double p, const int w)
	{
		assert((X_track.at(0).rows() == Y_track.at(0).at(0).rows()) && (X_track.at(0).cols() == Y_track.at(0).at(0).cols()));

		assert(c > 0);

		assert(p > 0);

		assert(w > 0);

		std::vector<Matrix> OSPA2_dist(N, Matrix(3, K));	// N = number of sensors, K = number of time steps

		std::vector<int> eval_idx(K);
		std::iota(eval_idx.begin(), eval_idx.end(), 0);	// Sets it to 0, ..., K-1

		bool truncated = true;

		std::vector<int> win_off(w);
		std::iota(win_off.begin(), win_off.end(), 1 - w);

		const int num_x = X_track.size();
		const int num_y = Y_track.at(0).size();

		const int num_step = K;
		const int num_eval = K;

		double val = std::pow(c, p);

		for (int n = 0; n < N; ++n)	// For each sensor
		{
			// First, for each time index, compute the matrix of inter-point distances and the track existence flags
			std::vector<Matrix> distances(num_step, Matrix(num_x, num_y));
			Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> x_exists(num_x, num_step);
			Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> y_exists(num_y, num_step);

			for (int k = 0; k < K; ++k)	// For each time step
			{
				// Compute distance between every pair of points
				std::vector<int> position_rows = { 0, 2 };	// Keeps the x and y position of tracks
				Matrix x(2, num_x);
				for (int t = 0; t < X_track.size(); ++t)
				{
					x.col(t) = X_track.at(t)(position_rows, k);	// Get the X_track for time step k, position only
				}
				Matrix y(2, num_y);
				for (int t = 0; t < X_track.size(); ++t)
				{
					y.col(t) = Y_track.at(n).at(t)(position_rows, k);	// Get the Y_track for sensor n and time step k, position only
				}
				Matrix d(num_x, num_y);
				d.setConstant(std::numeric_limits<double>::quiet_NaN());
				for (int j = 0; j < num_y; ++j)
				{
					// Broadcast subtraction of y.col(j) across all columns of x
					// Take absolute value, raise to power p, and sum down the rows (dimension axis)
					d.col(j) = (x.colwise() - y.col(j)).array().abs().pow(p).colwise().sum().transpose();
				}

				// Compute track existence flags
				x_exists.col(k) = !x.row(0).array().isNaN();
				y_exists.col(k) = !y.row(0).array().isNaN();

				// Distance between an empty and non-empty state
				Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> one_exists(num_x, num_y);
				for (int i = 0; i < num_x; ++i)
				{
					for (int j = 0; j < num_y; ++j)
					{
						one_exists(i, j) = (x_exists(i, k) != y_exists(j, k));
					}
				}

				for (int i = 0; i < d.rows(); ++i)
				{
					for (int j = 0; j < d.cols(); ++j)
					{
						if (one_exists(i, j))
						{
							d(i, j) = val;	// Replace distances between empty and non-empty state with c^p
						}
					}
				}

				// Find times when neither object exists
				Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> neither_exists(num_x, num_y);
				for (int i = 0; i < num_x; ++i)
				{
					for (int j = 0; j < num_y; ++j)
					{
						neither_exists(i, j) = (!x_exists(i, k) && !y_exists(j, k));
					}
				}
				if (truncated)
				{
					// Truncated window, exclude times when neither object exists
					for (int i = 0; i < d.rows(); ++i)
					{
						for (int j = 0; j < d.cols(); ++j)
						{
							if (neither_exists(i, j))
							{
								d(i, j) = NAN;	// Replace distances between empty and non-empty state with c^p
							}
						}
					}
				}

				// Store the distance matrix for this step
				distances.at(k) = d;
			}

			// Cap all inter-point distances at c^p
			if (truncated)
			{
				for (int k = 0; k < K; ++k)
				{
					for (int i = 0; i < distances.at(k).rows(); ++i)
					{
						for (int j = 0; j < distances.at(k).cols(); ++j)
						{
							// If either is NaN or greater than c^p, replace with c^p
							if (std::isnan(distances.at(k)(i, j)) || (distances.at(k)(i, j) > val))
							{
								distances.at(k)(i, j) = val;	// Replace distances that are either NaN or greater than c^p with c^p
							}
						}
					}
				}
			}

			// Compute the OSPA(2) at each evaluation point
			for (int i = 0; i < num_eval; ++i)
			{
				// Window indices
				std::vector<int> win_idx;

				for (int offset : win_off)
				{
					int k = i + offset;

					if (k >= 0 && k < num_step)
					{
						win_idx.push_back(k);
					}
				}

				// Get the number of objects in X and Y that exist at any time inside the current window
				std::vector<int> valid_rows;
				std::vector<int> valid_cols;

				for (int t = 0; t < num_x; ++t)
				{
					for (int k : win_idx)
					{
						if (x_exists(t, k))
						{
							valid_rows.push_back(t);
							break;
						}
					}
				}

				for (int t = 0; t < num_y; ++t)
				{
					for (int k : win_idx)
					{
						if (y_exists(t, k))
						{
							valid_cols.push_back(t);
							break;
						}
					}
				}

				const int m_tracks = static_cast<int>(valid_rows.size());
				const int n_tracks = static_cast<int>(valid_cols.size());

				double cost = 0.0;	// Initialize cost to 0

				// Compute the matrix of weighted time-averaged OSPA distances between tracks
				if (m_tracks > 0 && n_tracks > 0)
				{
					Matrix trk_dist(m_tracks, n_tracks);
					trk_dist.setZero();

					for (int a = 0; a < m_tracks; ++a)
					{
						for (int b = 0; b < n_tracks; ++b)
						{
							double sum = 0.0;
							int count = 0;

							// Find average distance between pairs of tracks over all time steps in window
							for (int k : win_idx)
							{
								const double d = distances.at(k)(valid_rows[a], valid_cols[b]);

								if (!std::isnan(d))
								{
									sum += d;
									++count;
								}
							}

							trk_dist(a, b) = (count > 0) ? (sum / count) : 0.0;	// Replace NaN scores with 0
						}
					}

					// Solve the optimal assignment problem
					if (m_tracks > n_tracks)
					{
						trk_dist.transposeInPlace();	// Transpose if necessary so rows <= columns
					}

					auto [assignment, assignment_cost] = solve_assignment(eigenToStdVector(trk_dist));

					cost = assignment_cost;
				}

				// Compute the OSPA(2) distance for the current time index
				const int max_tracks = std::max(m_tracks, n_tracks);

				if (max_tracks == 0)
				{
					OSPA2_dist.at(n).col(i).setZero();
				}
				else
				{
					const double cardinality_cost = val * std::abs(m_tracks - n_tracks);
					double dist = std::pow(1.0 / max_tracks * (cardinality_cost + cost), 1.0 / p);
					double loc = std::pow(1.0 / max_tracks * cost, 1.0 / p);
					double card = std::pow(1.0 / max_tracks * cardinality_cost, 1.0 / p);
					Vector result(3);
					result << dist, loc, card;
					OSPA2_dist.at(n).col(i) = result;
				}
			}
		}

		return OSPA2_dist;
	}
}