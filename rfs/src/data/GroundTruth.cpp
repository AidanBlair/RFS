#include <rfs/data/GroundTruth.hpp>
#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

using json = nlohmann::json;

namespace rfs
{
	static Matrix json_to_matrix(const json& j)
	{
		const int rows = j.size();
		const int cols = j.at(0).size();

		Matrix M(rows, cols);
		for (int r = 0; r < rows; ++r)
		{
			for (int c = 0; c < cols; ++c)
			{
				M(r, c) = j[r][c];
			}
		}

		return M;
	}

	GroundTruth GroundTruth::from_json(const std::string& filename)
	{
		std::ifstream f(filename);
		if (!f)
		{
			throw std::runtime_error("Could not open truth file!");
		}

		json j;
		f >> j;

		GroundTruth t;

		t.K = j.at("K").get<int>();
		t.total_tracks = j.at("total_tracks").get<int>();

		// X: vector of matrices
		for (const auto& Xi : j.at("X"))
		{
			t.X.push_back(json_to_matrix(Xi));
		}

		// N: vector<int>
		t.N = j.at("N").get<std::vector<int>>();

		// L and track_list: vector<vector<int>>
		t.L = j.at("L").get<std::vector<std::vector<int>>>();
		t.track_list = j.at("track_list").get<std::vector<std::vector<int>>>();

		// Size verification
		for (size_t k = 0; k < t.X.size(); ++k)
		{
			if (t.X[k].cols() != t.N[k])
			{
				throw std::runtime_error("X[k] columns != N[k]");
			}
			if (t.L[k].size() != static_cast<size_t>(t.N[k]))
			{
				throw std::runtime_error("L[k] size != N[k]");
			}
		}

		return t;
	}
}
