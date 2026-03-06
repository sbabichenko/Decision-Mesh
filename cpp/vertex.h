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
    bool active = false;
    bool disqualified = false;
    bool real_vertex = true;

    DecisionMesh* mesh = nullptr;
    Edge* parent_edge = nullptr;

    std::set<Edge*> edges;
    std::set<Vertex*> neighbors;
    std::set<Vertex*> affected_vertices;
    int affected_points = 0;

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
    };
    LocRegressResult loc_regress();

    int degree() const { return (int)edges.size(); }
};
