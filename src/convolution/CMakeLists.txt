set(target_prefix ${CMAKE_PROJECT_NAME})
set(target_name "image_convolution")
set(target_fullname "${target_prefix}_${target_name}")
set(taregt_source_files image_convolution.cpp)

# Let icpx find sycl includes.
set(target_compile_flags "-fsycl -Wall")
set(target_link_flags "-fsycl")
if (WIN32)
    set(target_compile_flags "${target_compile_flags} /EHsc")
endif ()

add_definitions(-Dworkspace_root="${CMAKE_SOURCE_DIR}/"
                -Dtarget_root="${CMAKE_CURRENT_LIST_DIR}/")   

add_executable(${target_fullname} ${taregt_source_files})
set_target_properties(${target_fullname} PROPERTIES COMPILE_FLAGS "${target_compile_flags}")
set_target_properties(${target_fullname} PROPERTIES LINK_FLAGS "${target_link_flags}")
add_custom_target("${target_fullname}_cpu-gpu" DEPENDS ${target_fullname})