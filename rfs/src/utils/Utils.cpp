#include "rfs/utils/Utils.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>

namespace rfs
{
	const GLMB castlmbpred(const std::vector<LMBComponent>& tt_lmb_birth, const std::vector<LMBComponent>& tt_lmb_survive, const int H_bth, const int H_sur)
	{
		// Creates GLMB prediction out of birth and survive densities
		// Express birth and surviving LMBs in GLMB structure
		GLMB glmb_birth = lmb2glmb(tt_lmb_birth, H_bth);
		GLMB glmb_survive = lmb2glmb(tt_lmb_survive, H_sur);

		// Generate predicted hypotheses/components (by convolution of birth and survive GLMB)
		GLMB glmb_predict;
		glmb_predict.tt = glmb_birth.tt;
		glmb_predict.tt.insert(glmb_predict.tt.end(), glmb_survive.tt.begin(), glmb_survive.tt.end());	// Adds glmb_survive.tt on to the end of glmb_birth.tt

		const int Nb = glmb_birth.w.size();
		const int Ns = glmb_survive.w.size();
		const int Nh = Nb * Ns;

		glmb_predict.w.resize(Nh);
		glmb_predict.n.resize(Nh);
		glmb_predict.I.resize(Nh);

		// Convolution of birth and survive hypotheses
		int hidx = 0;
		for (int bidx = 0; bidx < Nb; ++bidx)
		{
			for (int sidx = 0; sidx < Ns; ++sidx)
			{
				// Weight
				glmb_predict.w(hidx) = glmb_birth.w(bidx) * glmb_survive.w(sidx);

				// Label set
				glmb_predict.I[hidx] = glmb_birth.I[bidx];

				for (int label : glmb_survive.I[sidx])
				{
					glmb_predict.I[hidx].push_back(label + static_cast<int>(glmb_birth.tt.size()));
				}

				// Cardinality
				glmb_predict.n[hidx] = glmb_birth.n[bidx] + glmb_survive.n[sidx];

				++hidx;
			}
		}

		// Normalize weights
		double wsum = glmb_predict.w.sum();
		if (wsum > 0.0)
		{
			glmb_predict.w /= wsum;
		}

		// Extract cardinality distribution
		int max_cardinality = *std::max_element(glmb_predict.n.begin(), glmb_predict.n.end());
		glmb_predict.cdn = Eigen::VectorXd::Zero(max_cardinality + 1);

		for (int i = 0; i < glmb_predict.n.size(); ++i)
		{
			glmb_predict.cdn(glmb_predict.n[i]) += glmb_predict.w(i);
		}

		return glmb_predict;
	}

	const GLMB lmb2glmb(const std::vector<LMBComponent>& tt_lmb, const int H_req)
	{
		// Converts LMB density to GLMB density
		GLMB glmb;
		Vector rvect = get_rvals(tt_lmb);									// Vector of existence probabilities
		Vector costv = rvect.array() / (1.0 - rvect.array());						// Cost vector
		Vector neglogcostv = -costv.array().log();							// Negative log cost
		KShortestPathResult ksp = kshortestpath_pred(neglogcostv, H_req);	// k-shortest path to calculate k-best surviving hypotheses/components
		std::vector<std::vector<int>> paths = ksp.shortest_paths;
		std::vector<double> nlcost = ksp.total_costs;

		int num_hypotheses = nlcost.size();
		glmb.tt = tt_lmb;										// Track table
		glmb.I = paths;											// Tracks in hypothesis/component
		glmb.n.resize(num_hypotheses);
		glmb.w = Eigen::VectorXd::Zero(num_hypotheses);
		glmb.cdn = Eigen::VectorXd::Zero(num_hypotheses);
		for (int hidx = 0; hidx < num_hypotheses; ++hidx)
		{
			glmb.w.resize(num_hypotheses);
			double w = (1.0 - rvect.array()).log().sum() - nlcost[hidx];
			glmb.w(hidx) = w;									// Weight of hypothesis/component

			int n = paths[hidx].size();
			glmb.n[hidx] = n;								// Cardinality of hypothesis/component
		}
		double max_w = glmb.w.maxCoeff();
		double logsumexp = max_w + std::log((glmb.w.array() - max_w).exp().sum());
		glmb.w = (glmb.w.array() - logsumexp).exp();			// Normalize weights
		int max_cardinality = *std::max_element(glmb.n.begin(), glmb.n.end());
		glmb.cdn.resize(max_cardinality+1);
		for (int card = 0; card <= max_cardinality; ++card)
		{
			double sum = 0.0;
			for (int i = 0; i < glmb.n.size(); ++i)
			{
				if (glmb.n[i] == card)
				{
					sum += glmb.w(i);
				}
			}
			glmb.cdn(card) = sum;								// Cardinality distribution
		}

		return glmb;
	}

	const Vector get_rvals(const std::vector<LMBComponent>& tt_lmb)
	{
		// Function to extract vector of existence probabilities from LMB track table
		int len = tt_lmb.size();
		Vector rvect(len);
		for (int i = 0; i < len; ++i)
		{
			rvect(i) = tt_lmb[i].r;
		}
		return rvect;
	}

	const KShortestPathResult kshortestpath_pred(const Vector& rs, const int k)
	{
		KShortestPathResult result;

		if (k == 0)
		{
			return result;
		}

		const int ns = rs.size();

		// Sort weights in descending order
		std::vector<int> indices(ns);
		std::iota(indices.begin(), indices.end(), 0);	// Creates a vector of N = indices.end() - indices.begin() length, starting from 0, e.g. [0, 1, 2, ..., N-1]

		std::sort(indices.begin(), indices.end(), [&](int a, int b) { return rs(a) > rs(b); });	// lambda function for the comparator argument, if true then a comes before b. [&] means that by default it captures variables from surrounding scope by reference

		Vector ds(ns);
		for (int i = 0; i < ns; ++i)
		{
			ds(i) = rs(indices[i]);
		}

		// Cost matrix for paths (Directed Acyclic Graph)
		// REMEMBER ZERO COST DENOTES NO ARC CONNECTION, I.E. INF COST BECAUSE OF SPARSE MATRIX INPUT FOR KSHORTESTPATH
		Matrix CM = Eigen::MatrixXd::Zero(ns, ns);

		for (int i = 0; i < ns; ++i)
		{
			for (int j = 0; j < i; ++j)
			{
				CM(j, i) = ds(i);	// Only allow jumps to higher numbered nodes, inf costs (equivalent to zero cost for sparse representation) on lower diag prohibit reverse jumps (hence cycles)
			}
		}

		// Padded cost matrix
		Matrix CMPad = Eigen::MatrixXd::Zero(ns + 2, ns + 2);	// Extra 2 states for start and finish points

		// Start -> nodes
		CMPad.row(0).segment(1, ns) = ds.transpose();				// Must enter into one of the original nodes OR
		CMPad(0, ns + 1) = std::numeric_limits<double>::epsilon();	// Exit immediately indicating no node selection (all target die) (epsilon used to denote zero cost or free jump, as zero is used for inf in sparse format)

		// Nodes -> end
		CMPad.col(ns + 1).segment(1, ns).setConstant(std::numeric_limits<double>::epsilon());	// Must exit at last node at no cost (epsilon used to denote zero cost or free jump, as zero is used for inf in sparse format)

		// Internal edges
		CMPad.block(1, 1, ns, ns) = CM;	// Cost for original nodes

		// K Shortest Paths
		KShortestPathResult shortest_paths = kshortestpath_any(CMPad, 0, ns + 1, k);
		std::vector<std::vector<int>> raw_paths = shortest_paths.shortest_paths;
		std::vector<double> raw_costs = shortest_paths.total_costs;

		// Decode paths
		for (size_t p = 0; p < raw_paths.size(); ++p)
		{
			const auto& path = raw_paths[p];

			if (path.size() == 2)	// start -> end
			{
				result.shortest_paths.emplace_back();
			}
			else
			{
				std::vector<int> decoded;
				for (size_t i = 1; i + 1 < path.size(); ++i)
				{
					decoded.push_back(indices[path[i] - 1]);
				}
				
				result.shortest_paths.push_back(std::move(decoded));
			}

			result.total_costs.push_back(raw_costs[p]);
		}

		return result;
	}

	const KShortestPathResult kshortestpath_any(const Matrix& net_cost_matrix, const int source, const int destination, const int k_paths)
	{
		// Returns the k first shortest paths from source node to destination node using Yen's k-shortest path algorithm.
		KShortestPathResult result;

		int n = net_cost_matrix.rows();
		if (source >= n || destination >= n)	// The source or destination nodes are not part of net_cost_matrix
		{
			return result;	// Empty
		}

		// 1. Compute the first shortest path
		PathCost first = BFMSpath(net_cost_matrix, source, destination);
		if (first.path.empty())
		{
			return result;
		}

		std::vector<PathCost> P;		// All paths found
		std::vector<CandidatePath> X;	// Candidate paths
		std::vector<int> S;				// Deviation vertices

		P.push_back({ first.path, first.cost });
		S.push_back(first.path[0]);	// Deviation vertex is the first node initially
		X.push_back({ 0, first.path, first.cost });

		result.shortest_paths.push_back(first.path);			// The result of BFMSpath is the shortest path
		result.total_costs.push_back(first.cost);						// The result of BFMSpath is the shortest path

		int k = 1;
		int current_P = 0;

		// 2. Yen's k-shortest path loop
		while (k < k_paths && !X.empty())
		{
			// Remove current_P from X
			auto it = std::find_if(X.begin(), X.end(), [&](const CandidatePath& cp) { return cp.path_index == current_P; });
			if (it != X.end())
			{
				X.erase(it);
			}

			const auto P_ = P[current_P].path;

			// Find deciation vertex index
			int w = S[current_P];									// Deviation vertex (node number)
			auto w_it = std::find(P_.begin(), P_.end(), w);			// Iterator pointing to that node in the path
			int w_index_in_path = std::distance(P_.begin(), w_it);	// Index of that node in the path vector

			// Iterate over deviation vertices
			for (int index_dev_vertex = w_index_in_path; index_dev_vertex < (int)P_.size() - 1; ++index_dev_vertex)	// index_dev_vortex is index in P_ of deviation vertex
			{
				Matrix temp_net_cost_matrix = net_cost_matrix;

				// Remove vertices before deviation
				for (int i = 0; i < index_dev_vertex; ++i)
				{
					int v = P_[i];
					temp_net_cost_matrix.row(v).setConstant(std::numeric_limits<double>::infinity());
					temp_net_cost_matrix.col(v).setConstant(std::numeric_limits<double>::infinity());
				}

				// Remove edges that would recreate existing subpaths
				for (const auto& sp : result.shortest_paths)
				{
					if ((int)sp.size() > index_dev_vertex && std::equal(P_.begin(), P_.begin() + index_dev_vertex + 1, sp.begin()))	// Checks if sp is long enough to compare with the subpath in P_ and if they are equal
					{
						int next = sp[index_dev_vertex + 1];
						temp_net_cost_matrix(P_[index_dev_vertex], next) = std::numeric_limits<double>::infinity();
					}
				}

				// Cost of subpath before deviation
				double cost_sub_P = 0.0;
				for (int i = 0; i < index_dev_vertex; ++i)
				{
					cost_sub_P += net_cost_matrix(P_[i], P_[i + 1]);
				}

				// Shortest path from deviation to destination
				PathCost dev_result = BFMSpath(temp_net_cost_matrix, P_[index_dev_vertex], destination);

				if (!dev_result.path.empty())
				{
					// Concatenate subpath + dev path
					std::vector<int> new_path(P_.begin(), P_.begin() + index_dev_vertex);				// Create new_path with path from P_ up to index_dev_vertex
					new_path.insert(new_path.end(), dev_result.path.begin(), dev_result.path.end());	// Adds all elements from dev_result to end of new_path

					// Add to P
					P.push_back({ new_path, cost_sub_P + dev_result.cost });
					S.push_back(P_[index_dev_vertex]);

					// Add to candidate paths
					X.push_back({ static_cast<int>(P.size()) - 1, new_path, cost_sub_P + dev_result.cost });
				}
			}

			// Pick candidate with minimal cost
			if (!X.empty())
			{
				auto min_it = std::min_element(X.begin(), X.end(), [](const CandidatePath& a, const CandidatePath& b) { return a.cost < b.cost; });
				current_P = min_it->path_index;
				result.shortest_paths.push_back(P[current_P].path);
				result.total_costs.push_back(P[current_P].cost);
				++k;
			}
			else
			{
				break;
			}
		}

		return result;
	}

	const PathCost BFMSpath(const Matrix& G, const int source, const int destination)
	{
		// Bellman-Ford-Moore Shortest Path Algorithm
		int n = G.rows();
		std::vector<double> D(n, std::numeric_limits<double>::infinity());
		std::vector<int> p(n, -1);	// parent pointers, -1 = no parent (initialises every value of p to -1)

		D[source] = 0.0;

		// Bellman-Ford: relax edges up to n-1 times
		for (int iter = 0; iter < n; ++iter)
		{
			bool optimal = true;	// Track whether any relaxation occurs in this iteration
			for (int u = 0; u < n; ++u)
			{
				for (int v = u; v < n; ++v)
				{
					double weight = G(u, v);
					if (std::isfinite(weight) && D[u] + weight < D[v])
					{
						D[v] = D[u] + weight;	// Relax edge
						p[v] = u;				// Predecessor
						optimal = false;		// Some change occured
					}
				}
			}
			if (optimal)
			{
				break;	// Stop early if no changes occured
			}
		}

		// Build path from source to destination
		PathCost res;
		res.cost = D[destination];

		if (!std::isfinite(res.cost))
		{
			res.path.clear();	// No path
			return res;
		}

		std::vector<int> path;
		for (int v = destination; v != -1; v = p[v])	// Works backwards through the path from dstination -> source
		{
			path.push_back(v);
		}

		// Reverse to get path from source -> destination
		std::reverse(path.begin(), path.end());
		res.path = path;

		return res;
	}

	const std::vector<LMBComponent> glmb2lmb(const GLMB& glmb)
	{
		// Converts GLMB to LMB
		const int n_tracks = glmb.tt.size();

		LabelMatrix lmat(n_tracks, 3);

		for (int tabidx = 0; tabidx < n_tracks; ++tabidx)
		{
			lmat.row(tabidx) = glmb.tt[tabidx].l.transpose();
		}

		UniqueRowsResult urr = unique_rows_stable_label(lmat);
		LabelMatrix cu = urr.cu;
		std::vector<int> ic = urr.ic;

		// Preallocate tt_lmb
		const int n_unique = cu.rows();
		std::vector<LMBComponent> tt_lmb;
		tt_lmb.resize(n_unique);
		std::vector<int> num_particles(n_unique, 0);
		std::vector<int> particle_index(n_unique, 0);
		
		// Single pass to count particles
		int trkidx;
		int newidx;
		for (int hidx = 0; hidx < glmb.w.size(); ++hidx)
		{
			for (int t = 0; t < glmb.n[hidx]; ++t)
			{
				trkidx = glmb.I[hidx][t];
				newidx = ic[trkidx];
				num_particles[newidx] += glmb.tt[trkidx].x.cols();
			}
		}

		// Initialize structures
		for (int tabidx = 0; tabidx < n_unique; ++tabidx)
		{
			tt_lmb[tabidx].r = 0;
			tt_lmb[tabidx].l = cu.row(tabidx).transpose();
			tt_lmb[tabidx].x = Eigen::MatrixXd::Zero(5, num_particles[tabidx]);
			tt_lmb[tabidx].w = Eigen::VectorXd::Zero(num_particles[tabidx]);
		}

		// Extract individual tracks
		int start;
		int count;
		for (int hidx = 0; hidx < glmb.w.size(); ++hidx)	// Each hypothesis
		{
			for (int t = 0; t < glmb.n[hidx]; ++t)	// Each element in a hypothesis
			{
				trkidx = glmb.I[hidx][t];	// Element t of hypothesis
				newidx = ic[trkidx];
				start = particle_index[newidx];
				count = glmb.tt[trkidx].w.size();
				tt_lmb[newidx].x.middleCols(start, count) = glmb.tt[trkidx].x;
				tt_lmb[newidx].w.segment(start, count) = glmb.w(hidx) * glmb.tt[trkidx].w;
				particle_index[newidx] += glmb.tt[trkidx].w.size();
			}
		}

		// Extract existence probabilities and normalize track weights
		for (int tabidx = 0; tabidx < tt_lmb.size(); ++tabidx)
		{
			tt_lmb[tabidx].r = tt_lmb[tabidx].w.sum();
			tt_lmb[tabidx].w = tt_lmb[tabidx].w / tt_lmb[tabidx].r;
		}

		for (int tabidx = 0; tabidx < tt_lmb.size(); ++tabidx)
		{
			UniqueRowsResultXd temp = unique_rows_stable_numeric(tt_lmb[tabidx].x.transpose());
			Matrix xtemp = temp.cu.transpose();
			std::vector<int> icx = temp.ic;
			tt_lmb[tabidx].x = xtemp;
			tt_lmb[tabidx].w = accumarray(icx, tt_lmb[tabidx].w);
			tt_lmb[tabidx].w /= tt_lmb[tabidx].w.sum();
		}

		return tt_lmb;
	}

	const UniqueRowsResult unique_rows_stable_label(const LabelMatrix& lmat)
	{
		const int N = lmat.rows();

		std::unordered_map<Row3, int, Row3Hash> row_to_index;	// Hash table
		row_to_index.reserve(N);

		std::vector<Row3> unique_rows;	// Storage for unique rows
		unique_rows.reserve(N);

		std::vector<int> ic(N);	// Index map

		for (int i = 0; i < N; ++i)
		{
			Row3 r{ lmat(i, 0), lmat(i, 1), lmat(i, 2) };	// Extract row

			auto it = row_to_index.find(r);	// Check if row already exists
			if (it == row_to_index.end())	// If new row
			{
				int new_index = static_cast<int>(unique_rows.size());	// Assign next unique index
				row_to_index.emplace(r, new_index);	// Store mapping
				unique_rows.push_back(r);	// Append row in order
				ic[i] = new_index;	// Row i maps to its own new index
			}
			else
			{
				// Existing row
				ic[i] = it->second;	// Second value in the pair
			}
		}

		LabelMatrix cu(unique_rows.size(), 3);	// Convert back to LabelMatrix
		for (int i = 0; i < (int)unique_rows.size(); ++i)
		{
			cu(i, 0) = unique_rows[i].a;
			cu(i, 1) = unique_rows[i].b;
			cu(i, 2) = unique_rows[i].c;
		}

		return UniqueRowsResult{ cu, ic };
	}

	template <typename Derived> static const UniqueRowsResultXd unique_rows_stable_impl(const Eigen::MatrixBase<Derived>& Xexpr)
	{
		const auto& X = Xexpr.derived();	// Avoid copying the entire input matrix

		const int N = static_cast<int>(X.rows());	// Number of particles
		const int D = static_cast<int>(X.cols());	// Dimension of particles

		std::vector<Eigen::RowVectorXd> unique_rows;	// Unique particles
		unique_rows.reserve(N);

		std::vector<int> ic(N);	// Maps each input particle to its corresponding unique particle index

		std::unordered_map<std::size_t, std::vector<int>> buckets;	// Hash table: maps hash to indices of rows having that hash
		buckets.reserve(N);

		for (int i = 0; i < N; ++i)
		{
			// Hash the elements of the current row
			std::size_t hash = 0;

			for (int d = 0; d < D; ++d)
			{
				const std::size_t h = std::hash<double>{}(X(i, d));

				hash ^= h + 0x9e3779b9 + (hash << 6) + (hash >> 2);	// Mixes together
			}

			bool found = false;

			auto& candidates = buckets[hash];

			// Only compare rows with matching hashes
			for (int j : candidates)
			{
				if (X.row(i).isApprox(unique_rows[j], 0.0))
				{
					ic[i] = j;	// Record the existing corresponding unique particle index
					found = true;
					break;
				}
			}

			// If unique
			if (!found)
			{
				const int new_index = static_cast<int>(unique_rows.size());	// Set new index to the end of unique_rows

				unique_rows.emplace_back(X.row(i));	// Append the row to unique_rows
				candidates.push_back(new_index);	// Append the index to the correct bucket
				ic[i] = new_index;	// Record the new unique particle index
			}
		}

		Matrix cu(static_cast<Eigen::Index>(unique_rows.size()), D);

		for (int i = 0; i < static_cast<int>(unique_rows.size()); ++i)
		{
			cu.row(i) = unique_rows[i];
		}

		return UniqueRowsResultXd{ cu, ic };
	}

	const UniqueRowsResultXd unique_rows_stable_numeric(const Matrix& X)
	{
		return unique_rows_stable_impl(X);
	}

	const Vector accumarray(const std::vector<int>& ic, const Vector& w)
	{
		if (ic.empty())
		{
			return Vector(0);
		}

		const int N = ic.size();
		const int K = *std::max_element(ic.begin(), ic.end()) + 1;

		Vector w_new = Eigen::VectorXd::Zero(K);

		for (int i = 0; i < N; ++i)
		{
			w_new(ic[i]) += w(i);
		}

		return w_new;
	}

	// Estimates
	const Estimate extract_estimates(const std::vector<LMBComponent>& tt_lmb, const int x_dim, const double prediction_threshold)
	{
		Vector rvect = get_rvals(tt_lmb);

		int N = rvect.size();

		std::vector<int> idxkeep;
		idxkeep.reserve(N);

		for (int i = 0; i < N; ++i)
		{
			if (rvect(i) > prediction_threshold)
			{
				idxkeep.push_back(i);
			}
		}

		Vector rvect_filtered(idxkeep.size());
		for (int i = 0; i < static_cast<int>(idxkeep.size()); ++i)
		{
			rvect_filtered(i) = rvect(idxkeep[i]);
		}

		Matrix X = Eigen::MatrixXd::Zero(x_dim, N);
		Eigen::MatrixXi L = Eigen::MatrixXi::Zero(3, N);

		std::vector<int>idxcmp(N);
		std::iota(idxcmp.begin(), idxcmp.end(), 0);	// Assign increasing values starting from 0
		std::sort(idxcmp.begin(), idxcmp.end(), [&](int a, int b) { return rvect(a) > rvect(b); });	// descending

		for (int n = 0; n < N; ++n)
		{
			X.col(n) = tt_lmb[idxcmp[n]].x * tt_lmb[idxcmp[n]].w;	// EAP estimate
			L.col(n) = tt_lmb[idxcmp[n]].l;
		}

		Estimate est{X, N, L};
		return est;
	}

	// Measurements
	const Matrix gen_measurements(const DetectionModel& detection_model, const MeasurementModel& measurement_model, const Matrix& truth_X)
	{
		// Generate measurements
		Matrix observations = gen_observation(measurement_model, truth_X);											// Generate observations
		Matrix target_measurements = keep_observations_in_fov(detection_model, measurement_model, observations);	// Generate target measurements
		Matrix clutter_measurements = gen_clutter(detection_model, measurement_model);								// Generate clutter measurements
		Matrix measurements;																						// Concatenate both sets of measurements
		if (target_measurements.size() == 0 && clutter_measurements.size() == 0)
		{
			measurements.resize(0, 0);
		}
		else if (target_measurements.size() == 0)
		{
			measurements = clutter_measurements;
		}
		else if (clutter_measurements.size() == 0)
		{
			measurements = target_measurements;
		}
		else
		{
			measurements.resize(target_measurements.rows(), target_measurements.cols() + clutter_measurements.cols());
			measurements.leftCols(target_measurements.cols()) = target_measurements;
			measurements.rightCols(clutter_measurements.cols()) = clutter_measurements;
		}
		return measurements;
	}

	const Matrix gen_observation(const MeasurementModel& measurement_model, const Matrix& truth_X)
	{
		// Generate z_dimxN noisy observations based on ground truth
		Matrix Z(measurement_model.z_dim(), truth_X.size());

		if (truth_X.size() == 0)
		{
			return Z;
		}

		Matrix P = measurement_model.coordinate_extractor() * truth_X;	// Coordinate extraction

		Vector range_x = (P.row(0).transpose().array() - measurement_model.location()(0));
		Vector range_y = (P.row(1).transpose().array() - measurement_model.location()(1));
		Vector range = (range_x.array().square() + range_y.array().square()).sqrt();

		int count = 0;

		for (int i = 0; i < range.size(); ++i)
		{
			if (range(i) < measurement_model.range())
			{
				Vector sigma(measurement_model.z_dim());
				//sigma << 1 + (0.00001 * range_x(i) * range_x(i)), 1 + (0.00001 * range_y(i) * range_y(i));
				//sigma << 1 + (0.000001 * range_x(i) * range_x(i)), 1 + (0.000001 * range_y(i) * range_y(i));
				sigma << 0, 0;
				Vector V = sigma.array() * randn(measurement_model.z_dim()).array();
				Z.col(count) = P.col(i) + V;
				count++;
			}
		}

		Z.conservativeResize(Eigen::NoChange, count);

		return Z;
	}

	const Matrix keep_observations_in_fov(const DetectionModel& detection_model, const MeasurementModel& measurement_model, const Matrix& observations)
	{
		// Apply probability of detection to observations
		Matrix measurements(measurement_model.z_dim(), observations.cols());

		Matrix states = convert_observations_to_states(measurement_model, observations);
		Vector pD = detection_model.compute_pD(measurement_model, states, false);

		std::uniform_real_distribution<> uniform_dist(0, 1);
		int count = 0;

		for (int i = 0; i < states.cols(); ++i)
		{
			double r = uniform_dist(global_rng());
			if (r < pD(i))
			{
				measurements.col(count) = observations.col(i);
				count++;
			}
		}

		measurements.conservativeResize(Eigen::NoChange, count);

		return measurements;
	}

	const Matrix convert_observations_to_states(const MeasurementModel& measurement_model, const Matrix& observations)
	{
		// Converts z_dimxN observations to 5xN states
		Matrix states = Eigen::MatrixXd::Zero(measurement_model.x_dim(), observations.cols());

		states.row(0) = observations.row(0);
		states.row(2) = observations.row(1);

		return states;
	}

	const Matrix gen_clutter(const DetectionModel& detection_model, const MeasurementModel& measurement_model)
	{
		// Generate clutter measurements
		Matrix clutter_measurements;
		std::poisson_distribution<int> poisson_dist(measurement_model.clutter_intensity());
		int N_c = poisson_dist(global_rng());
		if (N_c == 0)
		{
			return clutter_measurements;
		}
		else
		{
			clutter_measurements.resize(measurement_model.z_dim(), N_c);
			Vector v(2);
			v << -1, 1;
			int count = 0;

			for (int i = 0; i < N_c; ++i)
			{
				Vector C = measurement_model.clutter_range().col(0) + (measurement_model.clutter_range() * v).asDiagonal() * rand(measurement_model.z_dim(), 0, 1);
				Vector C_ = convert_observations_to_states(measurement_model, C);
				//if (detection_model.compute_pD(measurement_model, C_, true)(0) == 1)
				//{
				//	clutter_measurements.col(count) = C;
				//	count++;
				//}
				clutter_measurements.col(count) = C;
				count++;
			}
			clutter_measurements.conservativeResize(Eigen::NoChange, count);
		}

		return clutter_measurements;
	}

	// Random
	std::mt19937& global_rng()
	{
		// Each thread gets its own generator
		//static thread_local std::mt19937 gen(std::random_device{}());
		static thread_local std::mt19937 gen(42);
		return gen;
	}

	Vector randn(const int N)
	{
		// Generates a Vector of size N that is normally distributed
		static thread_local std::normal_distribution<double> dist(0.0, 1.0);

		Vector v(N);
		for (int i = 0; i < N; ++i)
		{
			v(i) = dist(global_rng());
		}

		return v;
	}

	Vector rand(const int N, const int a, const int b)
	{
		// Generates a Vector of size N that is uniformly distributed between a and b
		static thread_local std::uniform_real_distribution<> dist(a, b);

		Vector v(N);
		for (int i = 0; i < N; ++i)
		{
			v(i) = dist(global_rng());
		}

		return v;
	}

	// Assignment
	MurtyResult mbestwrap_updt_custom(const std::vector<std::vector<double>>& P0, int m)
	{
		MurtyResult result;
		if (m == 0) return result;

		int n1 = P0.size();
		int n2 = P0[0].size();

		// ---- Pad with dummy columns ----
		std::vector<std::vector<double>> P = P0;
		for (int i = 0; i < n1; ++i) {
			for (int j = 0; j < n1; ++j) {
				P[i].push_back(0.0); // -log(1) = 0
			}
		}

		// ---- Make costs non-negative ----
		double x = std::numeric_limits<double>::infinity();
		for (auto& r : P)
			for (double v : r)
				x = std::min(x, v);

		for (auto& r : P)
		{
			for (double& v : r)
			{
				v -= x;
			}
		}

		// ---- Murty ----
		result = murty_custom(P, m);

		// ---- Restore true costs ----
		for (size_t k = 0; k < result.costs.size(); ++k) {
			int cnt = 0;
			for (int a : result.assignments[k])
				if (a >= 0) cnt++;
			result.costs[k] += x * cnt;
		}

		// ---- Strip dummy assignments ----
		for (auto& ass : result.assignments) {
			ass.resize(n1);
			for (int& a : ass)
				if (a >= n2) a = -1; // clutter
		}

		return result;
	}

	MurtyResult murty_custom(const std::vector<std::vector<double>>& P0, int m)
	{
		int num_rows = P0.size();
		int num_cols = P0[0].size();

		// ---- Initial solution ----
		auto [S0, C0] = solve_assignment(P0);

		MurtyResult out;
		if (m == 1) {
			out.assignments.push_back(S0);
			out.costs.push_back(C0);
			return out;
		}

		struct Node {
			std::vector<std::vector<double>> P;
			std::vector<int> S;
			double C;
		};

		std::vector<Node> queue;
		queue.push_back({ P0, S0, C0 });

		for (int k = 0; k < m; ++k) {

			if (queue.empty()) break;

			// ---- Pick lowest cost ----
			auto it = std::min_element(queue.begin(), queue.end(),
				[](const Node& a, const Node& b) {
					return a.C < b.C;
				});

			Node best = *it;
			queue.erase(it);

			out.assignments.push_back(best.S);
			out.costs.push_back(best.C);

			// ---- Branch ----
			for (int a = 0; a < num_rows; ++a) {
				int aj = best.S[a];
				if (aj < 0) continue;

				// ---- Remove assignment ----
				auto P_tmp = best.P;

				if (aj < num_cols - num_rows) {
					P_tmp[a][aj] = std::numeric_limits<double>::infinity();
				}
				else {
					for (int j = num_cols - num_rows; j < num_cols; ++j)
						P_tmp[a][j] = std::numeric_limits<double>::infinity();
				}

				auto [S_tmp, C_tmp] = solve_assignment(P_tmp);

				if (std::all_of(S_tmp.begin(), S_tmp.end(),
					[](int v) { return v >= 0; })) {
					queue.push_back({ P_tmp, S_tmp, C_tmp });
				}

				// ---- Enforce assignment ----
				auto P_enf = best.P;
				double v = P_enf[a][aj];
				for (int j = 0; j < num_cols; ++j) P_enf[a][j] = INFINITY;
				for (int i = 0; i < num_rows; ++i) P_enf[i][aj] = INFINITY;
				P_enf[a][aj] = v;
			}
		}

		return out;
	}

	std::pair<std::vector<int>, double> solve_assignment(const std::vector<std::vector<double>>& P)
	{
		// Performs the Hungarian Algorithm
		HungarianAlgorithm hungarian;
		std::vector<int> assignment;
		double cost = hungarian.Solve(const_cast<std::vector<std::vector<double>>&>(P), assignment);
		return { assignment, cost };
	}

	std::vector<std::vector<double>> eigenToStdVector(const Matrix& mat)
	{
		std::vector<std::vector<double>> result(mat.rows(), std::vector<double>(mat.cols()));

		for (int i = 0; i < mat.rows(); ++i)
			for (int j = 0; j < mat.cols(); ++j)
				result[i][j] = mat(i, j);

		return result;
	}

	// Pruning
	const std::vector<LMBComponent> clean_lmb(const std::vector<LMBComponent>& tt_lmb_in, const double track_threshold, const int T_max, const int number_particles_track)
	{
		for (int tabidx = 0; tabidx < tt_lmb_in.size(); ++tabidx)
		{
			if (tt_lmb_in[tabidx].x.cols() > 0)
			{
				auto mean_state = [](const Matrix& x, const Vector& w)
					{
						return x * w;
					};

				Vector mean_first = mean_state(tt_lmb_in[tabidx].x, tt_lmb_in[tabidx].w);

				const auto& w = tt_lmb_in[tabidx].w;
			}
		}
		
		// Prune tracks with low existence probabilities
		Vector rvect = get_rvals(tt_lmb_in);

		std::vector<LMBComponent> tt_lmb_out;
		std::vector<double> rvals_out;

		tt_lmb_out.reserve(tt_lmb_in.size());
		rvals_out.reserve(tt_lmb_in.size());

		for (int i = 0; i < rvect.size(); ++i)
		{
			if (rvect(i) > track_threshold)
			{
				tt_lmb_out.push_back(tt_lmb_in[i]);
				rvals_out.push_back(rvect(i));
			}
		}

		// Enforce cap on maximum number of tracks
		if (tt_lmb_out.size() > T_max)
		{
			std::sort(tt_lmb_out.begin(), tt_lmb_out.end(), [](const LMBComponent& a, const LMBComponent& b) {return a.r > b.r; });	// Descending existence probability

			tt_lmb_out.resize(T_max);
		}

		// Cleanup tracks
		for (int tabidx = 0; tabidx < tt_lmb_out.size(); ++tabidx)
		{
			Matrix xtemptemp = tt_lmb_out[tabidx].x;
			Vector wtemptemp = tt_lmb_out[tabidx].w;

			if ((wtemptemp.array() < 0.0).any() || wtemptemp.sum() <= 0.0)
			{
				std::cerr << "W must contain non-negative values with at least one positive value.\n";
				continue;
			}

			Vector wnorm = wtemptemp / wtemptemp.sum();	// Normalize weights

			//std::discrete_distribution<int> dist(wnorm.data(), wnorm.data() + wnorm.size());	// Weighted random sampling with replacement, Pointers to start and end+1 of wnorm in memory
			//std::vector<int> rspidx(number_particles_track);
			//for (int i = 0; i < number_particles_track; ++i)
			//{
			//	rspidx[i] = dist(global_rng());
			//}
			std::vector<int> rspidx = systematic_resample(wnorm, number_particles_track);

			Matrix xnew(xtemptemp.rows(), number_particles_track);
			for (int i = 0; i < number_particles_track; ++i)
			{
				xnew.col(i) = xtemptemp.col(rspidx[i]);	// Resample particles
			}

			tt_lmb_out[tabidx].x = xnew;
			tt_lmb_out[tabidx].w = Eigen::VectorXd::Constant(number_particles_track, 1.0 / number_particles_track);	// Reset weights

			auto mean_state = [](const Matrix& x, const Vector& w)
				{
					return x * w;
				};

			Vector mean_before = mean_state(xtemptemp, wnorm);
			Vector mean_after = mean_state(tt_lmb_out[tabidx].x, tt_lmb_out[tabidx].w);
		}

		return tt_lmb_out;
	}

	// Proper randsample
	std::vector<int> systematic_resample(const Eigen::VectorXd& w, const int N)
	{
		//int N = w.size();
		std::vector<int> idx(N);

		Eigen::VectorXd cdf = w;
		for (int i = 1; i < N; ++i)
			cdf(i) += cdf(i - 1);

		std::uniform_real_distribution<double> uni(0.0, 1.0 / N);
		double u0 = uni(global_rng());

		int i = 0;
		for (int j = 0; j < N; ++j)
		{
			double uj = u0 + j * (1.0 / N);
			while (uj > cdf(i))
				++i;
			idx[j] = i;
		}
		return idx;
	}

	// Metric helpers
	std::vector<Matrix> extract_tracks(const std::vector<Matrix>& X, const std::vector<std::vector<int>>& track_list, const int total_tracks, const int K)
	{
		int x_dim = X.at(K - 1).rows();
		int k = K - 2;
		while (x_dim == 0)
		{
			x_dim = X.at(k).rows();
			k--;
		}

		std::vector<Matrix> X_track(total_tracks);

		for (int t = 0; t < total_tracks; ++t)
		{
			X_track.at(t) = Matrix::Constant(x_dim, K, std::numeric_limits<double>::quiet_NaN());

			for (int k = 0; k < K; ++k)
			{
				for (int j = 0; j < static_cast<int>(track_list.at(k).size()); ++j)
				{
					if ((track_list.at(k).at(j) - 1) == t)	// Track list starts at 1
					{
						X_track.at(t).col(k) = X.at(k).col(j);
						break;
					}
				}
			}
		}

		return X_track;
	}

	int countestlabels(Estimates& est, const int K)
	{
		// Measure the total number of labels
		int total_cols = 0;
		for (int k = 0; k < K; ++k)
		{
			total_cols += est.L(k).cols();
		}

		// Create matrix to store all labels
		Matrix labelstack(3, total_cols);

		// Populate matrix
		int current_col = 0;
		for (int k = 0; k < K; ++k)
		{
			int ncols = est.L(k).cols();
			labelstack.middleCols(current_col, ncols) = est.L(k).cast<double>();
			current_col += ncols;
		}

		// Number of unique labels
		int count = unique_rows_stable_numeric(labelstack.transpose()).cu.rows();
		return count;
	}

	ColorArray makecolorarray(int labelcount)
	{
		ColorArray colorarray;
		colorarray.rgb.reserve(labelcount);
		colorarray.label_to_idx.reserve(labelcount);
		return colorarray;
	}

	void assigncolorarray(Estimates& est, const GroundTruth& truth, int labelcount)
	{
		std::uniform_real_distribution<double> dist(0.1, 0.9);

		ColorArray colorarray = makecolorarray(labelcount);

		for (int k = 0; k < truth.K; ++k)
		{
			const auto& X = est.X(k);
			const auto& L = est.L(k);

			std::vector<int> track_list_k;
			track_list_k.reserve(X.cols());

			for (Eigen::Index eidx = 0; eidx < X.cols(); ++eidx)
			{
				Row3 label{
					L(0, eidx),
					L(1, eidx),
					L(2, eidx)
				};

				auto it = colorarray.label_to_idx.find(label);
				int idx;

				if (it != colorarray.label_to_idx.end())
				{
					// Label already has a color
					idx = it->second;
				}
				else
				{
					// Assign a new color to this label
					idx = static_cast<int>(colorarray.rgb.size());

					colorarray.label_to_idx.emplace(label, idx);
					colorarray.rgb.push_back({
						dist(global_rng()),
						dist(global_rng()),
						dist(global_rng())
						});
				}
				
				track_list_k.push_back(idx + 1);	// Start at 1
			}

			est.set_track_list(k, track_list_k);
		}
	}
}