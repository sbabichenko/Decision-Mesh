cmake_minimum_required(VERSION 3.14)
project(DecisionMesh CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(decision_mesh
    main.cpp
    mesh.cpp
    vertex.cpp
    edge.cpp
    face.cpp
    tree.cpp
)

target_compile_options(decision_mesh PRIVATE -O2)
