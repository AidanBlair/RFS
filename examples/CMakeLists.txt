add_executable(lmb_demo
	lmb_demo.cpp
)

target_link_libraries(lmb_demo
	PRIVATE
		rfs
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/data/test_ground_truth.json"
    "${CMAKE_CURRENT_BINARY_DIR}/test_ground_truth.json"
    COPYONLY
)