#include "face.h"
#include "mesh.h"
#include "edge.h"
#include "vertex.h"
#include "tree.h"
#include <cmath>
#include <cstring>
#include <algorithm>

int Face::_seq = 0;

Face::Face(DecisionMesh* mesh, Edge* e0, Edge* e1, Edge* e2,
           const std::vector<bool>& mask, bool active_flag,
           const std::string& path, bool skip_coords)
    : _id(Face::_seq++), path(path), mesh(mesh), mask(mask)
{
    edges[0] = e0;
    edges[1] = e1;
    edges[2] = e2;

    // Resolve vertices: vertex0 is opposite edge0
    Vertex* v1 = e0->vertex0;
    Vertex* v2 = e0->vertex1;
    Vertex* v0 = e2->other_vertex(v1);
    if (!v0) {
        std::swap(v1, v2);
        v0 = e2->other_vertex(v1);
    }
    vertices[0] = v0;
    vertices[1] = v1;
    vertices[2] = v2;

    if (!skip_coords) {
        update_coords();
    }

    if (active_flag) {
        activate();
    }
}

std::string Face::sid() const {
    char buf[16];
    snprintf(buf, sizeof(buf), "F%03d", _id);
    return buf;
}

void Face::activate() {
    if (active) return;
    active = true;
    mesh->active_faces.insert(this);
    for (int i = 0; i < 3; ++i) {
        edges[i]->activate();
        edges[i]->add_face(this);
    }
}

void Face::deactivate() {
    if (!active) return;
    active = false;
    mesh->active_faces.erase(this);
    for (int i = 0; i < 3; ++i) {
        edges[i]->remove_face(this);
    }
}

void Face::split(Edge* edge) {
    if (edge != edges[0]) {
        // Completion: must split along refinement edge first
        edges[0]->midpoint->activate();
        return;
    }

    auto& sd = sub_divisions[0];
    if (!sd.e) {
        return;  // Edge not activated yet
    }

    deactivate();
    node->split(edge, sd.plus_face, sd.minus_face);
    sd.plus_face->activate();
    sd.minus_face->activate();
}

double Face::area() const {
    double det = (vertices[0]->x - vertices[2]->x) * (vertices[1]->y - vertices[0]->y) -
                 (vertices[0]->x - vertices[1]->x) * (vertices[2]->y - vertices[0]->y);
    return 0.5 * std::abs(det);
}

double Face::aspect_ratio() const {
    double max_len = 0, min_len = 1e300;
    for (int i = 0; i < 3; ++i) {
        double l = edges[i]->length;
        if (l > max_len) max_len = l;
        if (l < min_len) min_len = l;
    }
    if (min_len <= 0) return 1e300;
    return max_len / min_len;
}

// Accumulate sufficient statistics for one point with barycentric weights w
// and response value y. Inlined to avoid function call overhead in hot loop.
static inline void accum_suff_stats(double S_ww[3][3], double S_wy[3],
                                     double& S_yy, double w0, double u, double v,
                                     double y) {
    // S_ww: only upper triangle, symmetric fill at end
    S_ww[0][0] += w0 * w0;
    S_ww[0][1] += w0 * u;
    S_ww[0][2] += w0 * v;
    S_ww[1][1] += u * u;
    S_ww[1][2] += u * v;
    S_ww[2][2] += v * v;
    S_wy[0] += w0 * y;
    S_wy[1] += u * y;
    S_wy[2] += v * y;
    S_yy += y * y;
}

static inline void fill_symmetric(double S_ww[3][3]) {
    S_ww[1][0] = S_ww[0][1];
    S_ww[2][0] = S_ww[0][2];
    S_ww[2][1] = S_ww[1][2];
}

void Face::update_coords(double eps) {
    coords_indices.clear();

    // Zero sufficient statistics
    std::memset(S_ww, 0, sizeof(S_ww));
    std::memset(S_wy, 0, sizeof(S_wy));
    S_yy = 0.0;
    n_covered = 0;

    Vertex* v0 = vertices[0];
    Vertex* v1 = vertices[1];
    Vertex* v2 = vertices[2];

    // Right-triangle property: v0 is the right-angle vertex (opposite hypotenuse).
    // Legs a = v1-v0, b = v2-v0 are perpendicular, so a·b = 0.
    // Barycentric coords simplify: u = (p·a)/|a|², v = (p·b)/|b|²
    double ax = v1->x - v0->x, ay = v1->y - v0->y;
    double bx = v2->x - v0->x, by = v2->y - v0->y;

    double d00 = ax * ax + ay * ay;
    double d11 = bx * bx + by * by;

    bool degenerate = (d00 < eps || d11 < eps);

    if (!degenerate) {
        double inv_d00 = 1.0 / d00;
        double inv_d11 = 1.0 / d11;
        double v0x = v0->x, v0y = v0->y;

        for (int i = 0; i < mesh->n_points; ++i) {
            if (!mask[i]) continue;

            double px = mesh->get_x(i, 0) - v0x;
            double py = mesh->get_x(i, 1) - v0y;
            double u = (px * ax + py * ay) * inv_d00;
            double v = (px * bx + py * by) * inv_d11;
            double w0 = 1.0 - u - v;

            coords_indices.push_back(i);
            accum_suff_stats(S_ww, S_wy, S_yy, w0, u, v, mesh->values[i]);
            n_covered++;
        }
    } else {
        for (int i = 0; i < mesh->n_points; ++i) {
            if (!mask[i]) continue;
            coords_indices.push_back(i);
        }
    }

    fill_symmetric(S_ww);
}

void Face::update_coords_from_indices(const std::vector<int>& indices, double eps) {
    coords_indices.clear();

    std::memset(S_ww, 0, sizeof(S_ww));
    std::memset(S_wy, 0, sizeof(S_wy));
    S_yy = 0.0;
    n_covered = 0;

    Vertex* v0 = vertices[0];
    Vertex* v1 = vertices[1];
    Vertex* v2 = vertices[2];

    double ax = v1->x - v0->x, ay = v1->y - v0->y;
    double bx = v2->x - v0->x, by = v2->y - v0->y;

    double d00 = ax * ax + ay * ay;
    double d11 = bx * bx + by * by;

    bool degenerate = (d00 < eps || d11 < eps);

    coords_indices.reserve(indices.size());

    if (!degenerate) {
        double inv_d00 = 1.0 / d00;
        double inv_d11 = 1.0 / d11;
        double v0x = v0->x, v0y = v0->y;

        for (int i : indices) {
            double px = mesh->get_x(i, 0) - v0x;
            double py = mesh->get_x(i, 1) - v0y;
            double u = (px * ax + py * ay) * inv_d00;
            double v = (px * bx + py * by) * inv_d11;
            double w0 = 1.0 - u - v;

            coords_indices.push_back(i);
            accum_suff_stats(S_ww, S_wy, S_yy, w0, u, v, mesh->values[i]);
            n_covered++;
        }
    } else {
        for (int i : indices) {
            coords_indices.push_back(i);
        }
    }

    fill_symmetric(S_ww);
}

void Face::add_sub_division(Edge* edge, SubDivision sd) {
    for (int i = 0; i < 3; ++i) {
        if (edges[i] == edge) {
            sub_divisions[i] = sd;
            return;
        }
    }
}
