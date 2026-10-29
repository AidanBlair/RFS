add_library(rfs STATIC)

target_sources(rfs
	PRIVATE
		src/lmb/LMBFilter.cpp
		src/models/MotionModel.cpp
		src/models/MeasurementModel.cpp
		src/models/BirthModel.cpp
		src/models/DetectionModel.cpp
		src/models/NetworkModel.cpp
		src/data/GroundTruth.cpp
		src/data/Estimates.cpp
		src/data/Measurements.cpp
		src/utils/Utils.cpp
		src/utils/Hungarian.cpp
		src/utils/Metrics.cpp
		
	PUBLIC
		include/rfs/lmb/LMBFilter.hpp
		include/rfs/lmb/LMBComponent.hpp
		include/rfs/glmb/GLMB.hpp
		include/rfs/models/MotionModel.hpp
		include/rfs/models/MeasurementModel.hpp
		include/rfs/models/BirthModel.hpp
		include/rfs/models/DetectionModel.hpp
		include/rfs/models/NetworkModel.hpp
		include/rfs/data/GroundTruth.hpp
		include/rfs/data/Estimates.hpp
		include/rfs/data/Measurements.hpp
		include/rfs/utils/Utils.hpp
		include/rfs/utils/Hungarian.hpp
		include/rfs/utils/Metrics.hpp
 )
 
 add_library(RFS::rfs ALIAS rfs)

target_include_directories(rfs
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(rfs
    PUBLIC
        Eigen3::Eigen
        nlohmann_json::nlohmann_json
)