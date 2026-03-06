#pragma once
#include <cmath>
#include <set>
#include <string>
#include <vector>

struct DecisionMesh;
struct Edge;
struct Face;

struct Vertex {
    static int _seq;

    int _id;
    double x, y;
    double height = 0.0;
    double new_height = 0.0;
    double loss_reduction = 0.0;
    double heap_key = 0.0;       // current key in loss_heap (for O(log n) removal)
    bool in_heap = false;        // whether this vertex is in the heap
    bool active = false;
    bool disqualified = false;
    bool real_vertex = true;

    DecisionMesh* mesh = nullptr;
    Edge* parent_edge = nullptr;

    std::set<Edge*> edges;
    std::set<Vertex*> neighbors;
    std::set<Vertex*> affected_vertices;
    int affected_points = 0;

    // Empirical Bayes partial pooling fields
    int depth = 0;
    double sigma_sq = 1e300;       // data-only variance (1/xTx)
    double delta_pooled = 0.0;     // height - mu_lin
    double sigma_pooled = 1e300;   // posterior variance
    double lambda_v = 0.0;        // prior precision (1/sigma_prior)
    double prior_mean = 0.0;      // EB prior mean
    std::set<Vertex*> prior_children;  // vertices on this vertex's edges (for prior propagation)

    Vertex() : _id(-1), x(0), y(0) {}
    Vertex(DecisionMesh* mesh, double x, double y, bool active = false,
           Edge* parent_edge = nullptr, bool real_vertex = true);

    std::string sid() const;

    // Vector arithmetic (returns temp vertex, not registered)
    double norm() const { return std::hypot(x, y); }
    double dot(const Vertex& o) const { return x * o.x + y * o.y; }
    double dot2(double a, double b) const { return x * a + y * b; }

    void add_edge(Edge* e);
    void remove_edge(Edge* e);

    void activate();
    void update_info();
    void update_height();

    // Face collection
    struct FacesResult {
        std::set<Vertex*> neighbors;
        std::set<Face*> faces;
    };
    FacesResult get_faces() const;
    FacesResult get_sim_faces() const;

    std::set<Vertex*> neighbors_after_split() const;

    // Local regression
    struct LocRegressResult {
        double beta_opt;
        double loss_reduction;
        std::set<Vertex*> affected;
        int n_points;
        double xTx, xTr, rTr;  // sufficient statistics for EB
    };
    LocRegressResult loc_regress();

    // Empirical Bayes helpers
    double mu_lin() const;
    double compute_overlap_factor(Vertex* a, Vertex* b) const;
    std::pair<double, double> compute_eb_params(double xTx) const;

    int degree() const { return (int)edges.size(); }
};
