#include "face.h"
#include "mesh.h"
#include "edge.h"
#include "vertex.h"
#include "tree.h"
#include <cmath>
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

void Face::update_coords(double eps) {
    coords_indices.clear();
    coords_weights.clear();

    // Zero out sufficient statistics
    for (int i = 0; i < 3; ++i) {
        S_wy[i] = 0.0;
        for (int j = 0; j < 3; ++j)
            S_ww[i][j] = 0.0;
    }
    S_yy = 0.0;
    n_covered = 0;

    Vertex* v0 = vertices[0];
    Vertex* v1 = vertices[1];
    Vertex* v2 = vertices[2];

    // a = v1 - v0, b = v2 - v0
    double ax = v1->x - v0->x, ay = v1->y - v0->y;
    double bx = v2->x - v0->x, by = v2->y - v0->y;

    double d00 = ax * ax + ay * ay;
    double d01 = ax * bx + ay * by;
    double d11 = bx * bx + by * by;
    double denom = d00 * d11 - d01 * d01;

    bool degenerate = (std::abs(denom) < eps);

    for (int i = 0; i < mesh->n_points; ++i) {
        if (!mask[i]) continue;

        double px = mesh->get_x(i, 0) - v0->x;
        double py = mesh->get_x(i, 1) - v0->y;

        if (degenerate) {
            coords_indices.push_back(i);
            coords_weights.push_back({NAN, NAN, NAN});
        } else {
            double d20 = px * ax + py * ay;
            double d21 = px * bx + py * by;
            double u = (d11 * d20 - d01 * d21) / denom;
            double v = (d00 * d21 - d01 * d20) / denom;
            double w0 = 1.0 - u - v;
            coords_indices.push_back(i);
            coords_weights.push_back({w0, u, v});

            // Accumulate sufficient statistics
            double w[3] = {w0, u, v};
            double y = mesh->values[i];
            for (int a = 0; a < 3; ++a) {
                S_wy[a] += w[a] * y;
                for (int b = a; b < 3; ++b) {
                    S_ww[a][b] += w[a] * w[b];
                }
            }
            S_yy += y * y;
            n_covered++;
        }
    }

    // Fill symmetric entries
    S_ww[1][0] = S_ww[0][1];
    S_ww[2][0] = S_ww[0][2];
    S_ww[2][1] = S_ww[1][2];
}

void Face::update_coords_from_indices(const std::vector<int>& indices, double eps) {
    coords_indices.clear();
    coords_weights.clear();

    for (int i = 0; i < 3; ++i) {
        S_wy[i] = 0.0;
        for (int j = 0; j < 3; ++j)
            S_ww[i][j] = 0.0;
    }
    S_yy = 0.0;
    n_covered = 0;

    Vertex* v0 = vertices[0];
    Vertex* v1 = vertices[1];
    Vertex* v2 = vertices[2];

    double ax = v1->x - v0->x, ay = v1->y - v0->y;
    double bx = v2->x - v0->x, by = v2->y - v0->y;

    double d00 = ax * ax + ay * ay;
    double d01 = ax * bx + ay * by;
    double d11 = bx * bx + by * by;
    double denom = d00 * d11 - d01 * d01;

    bool degenerate = (std::abs(denom) < eps);

    coords_indices.reserve(indices.size());
    coords_weights.reserve(indices.size());

    for (int i : indices) {
        double px = mesh->get_x(i, 0) - v0->x;
        double py = mesh->get_x(i, 1) - v0->y;

        if (degenerate) {
            coords_indices.push_back(i);
            coords_weights.push_back({NAN, NAN, NAN});
        } else {
            double d20 = px * ax + py * ay;
            double d21 = px * bx + py * by;
            double u = (d11 * d20 - d01 * d21) / denom;
            double v = (d00 * d21 - d01 * d20) / denom;
            double w0 = 1.0 - u - v;
            coords_indices.push_back(i);
            coords_weights.push_back({w0, u, v});

            double w[3] = {w0, u, v};
            double y = mesh->values[i];
            for (int a = 0; a < 3; ++a) {
                S_wy[a] += w[a] * y;
                for (int b = a; b < 3; ++b) {
                    S_ww[a][b] += w[a] * w[b];
                }
            }
            S_yy += y * y;
            n_covered++;
        }
    }

    S_ww[1][0] = S_ww[0][1];
    S_ww[2][0] = S_ww[0][2];
    S_ww[2][1] = S_ww[1][2];
}

void Face::add_sub_division(Edge* edge, SubDivision sd) {
    for (int i = 0; i < 3; ++i) {
        if (edges[i] == edge) {
            sub_divisions[i] = sd;
            return;
        }
    }
}
