#pragma once
#include <array>
#include <string>
#include <vector>
#include <map>

struct DecisionMesh;
struct Edge;
struct Vertex;
struct TreeNode;

struct Face;

struct SubDivision {
    Edge* e = nullptr;
    Face* plus_face = nullptr;
    Face* minus_face = nullptr;
};

struct Face {
    static int _seq;

    int _id;
    std::string path;
    DecisionMesh* mesh = nullptr;
    TreeNode* node = nullptr;
    bool active = false;

    std::array<Edge*, 3> edges = {nullptr, nullptr, nullptr};
    std::array<Vertex*, 3> vertices = {nullptr, nullptr, nullptr};
    std::array<SubDivision, 3> sub_divisions;

    // mask: indices of data points in this face
    std::vector<bool> mask;

    // Barycentric coordinates: for each masked point, weights [w0, w1, w2]
    // coords_indices[i] = index into global data array
    // coords_weights[i] = {w0, w1, w2} for vertices[0..2]
    std::vector<int> coords_indices;
    std::vector<std::array<double, 3>> coords_weights;

    // Sufficient statistics for regression (computed once in update_coords)
    // S_ww[i][j] = Σ w_i * w_j over all points in face
    // S_wy[i]    = Σ w_i * y   over all points in face
    // S_yy       = Σ y²        over all points in face
    double S_ww[3][3] = {};
    double S_wy[3] = {};
    double S_yy = 0.0;
    int n_covered = 0;

    Face() : _id(-1) {}
    Face(DecisionMesh* mesh, Edge* e0, Edge* e1, Edge* e2,
         const std::vector<bool>& mask, bool active = true,
         const std::string& path = "", bool skip_coords = false);

    std::string sid() const;

    Edge* refinement_edge() const { return edges[0]; }

    void activate();
    void deactivate();
    void split(Edge* edge);

    double area() const;
    double aspect_ratio() const;

    void update_coords(double eps = 1e-14);
    // Fast path: compute coords only for given indices (subset of parent face)
    void update_coords_from_indices(const std::vector<int>& indices, double eps = 1e-14);
    void add_sub_division(Edge* edge, SubDivision sd);
    void addnode(TreeNode* n) { node = n; }
};
