#pragma once
#include <array>
#include <map>
#include <set>
#include <string>

struct DecisionMesh;
struct Vertex;
struct Face;

struct Edge {
    static int _seq;

    int _id;
    Vertex* vertex0 = nullptr;
    Vertex* vertex1 = nullptr;
    DecisionMesh* mesh = nullptr;

    double normal[2] = {0, 0};
    double length = 0.0;
    double intercept = 0.0;
    bool active = false;

    Vertex* midpoint = nullptr;

    // '+' = index 0, '-' = index 1
    Face* faces[2] = {nullptr, nullptr};
    Vertex* opposing_vertices[2] = {nullptr, nullptr};

    // sub_edges: '0'=0, '1'=1, '+'=2, '-'=3
    Edge* sub_edges[4] = {nullptr, nullptr, nullptr, nullptr};

    std::set<Face*> disqualifying;

    Edge() : _id(-1) {}
    Edge(DecisionMesh* mesh, Vertex* v0, Vertex* v1, bool active);

    std::string sid() const;

    static int sign_index(char s) { return s == '+' ? 0 : 1; }

    void activate();
    void add_midpoint();
    void split();
    void deactivate();

    double test_vertex(Vertex* v) const;
    Vertex* other_vertex(Vertex* v) const;

    void add_face(Face* face);
    void remove_face(Face* face);
};
