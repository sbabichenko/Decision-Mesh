#include "edge.h"
#include "mesh.h"
#include "vertex.h"
#include "face.h"
#include <cmath>
#include <algorithm>

int Edge::_seq = 0;

Edge::Edge(DecisionMesh* mesh, Vertex* v0, Vertex* v1, bool active_flag)
    : _id(Edge::_seq++), vertex0(v0), vertex1(v1), mesh(mesh)
{
    double dx = v1->x - v0->x;
    double dy = v1->y - v0->y;
    double n0 = -dy, n1 = dx;
    length = std::hypot(n0, n1);
    if (length > 0) {
        normal[0] = n0 / length;
        normal[1] = n1 / length;
    }
    intercept = normal[0] * v0->x + normal[1] * v0->y;

    if (active_flag) {
        activate();
    }
}

std::string Edge::sid() const {
    char buf[16];
    snprintf(buf, sizeof(buf), "E%03d", _id);
    return buf;
}

void Edge::activate() {
    if (active) return;
    active = true;
    mesh->active_edges.insert(this);
    add_midpoint();
    vertex0->add_edge(this);
    vertex1->add_edge(this);

    // Create half sub-edges (inactive)
    sub_edges[0] = mesh->make_edge(vertex0, midpoint, false);  // '0'
    sub_edges[1] = mesh->make_edge(midpoint, vertex1, false);  // '1'
}

void Edge::add_midpoint() {
    if (midpoint) return;
    double mx = (vertex0->x + vertex1->x) / 2.0;
    double my = (vertex0->y + vertex1->y) / 2.0;
    midpoint = mesh->make_vertex(mx, my, false, this, true);
}

void Edge::split() {
    // Split faces on both sides
    for (int s = 0; s < 2; ++s) {
        while (faces[s] != nullptr) {
            faces[s]->split(this);
        }
    }

    sub_edges[0]->activate();  // '0'
    sub_edges[1]->activate();  // '1'
    deactivate();
}

void Edge::deactivate() {
    if (!active) return;
    active = false;
    mesh->active_edges.erase(this);
    vertex0->remove_edge(this);
    vertex1->remove_edge(this);
}

double Edge::test_vertex(Vertex* v) const {
    return normal[0] * v->x + normal[1] * v->y - intercept;
}

Vertex* Edge::other_vertex(Vertex* v) const {
    if (v == vertex0) return vertex1;
    if (v == vertex1) return vertex0;
    return nullptr;
}

void Edge::add_face(Face* face) {
    // Find opposing vertex
    int idx = -1;
    for (int i = 0; i < 3; ++i) {
        if (face->edges[i] == this) { idx = i; break; }
    }
    Vertex* opposing = face->vertices[idx];
    int si = (test_vertex(opposing) > 0) ? 0 : 1;  // 0='+', 1='-'
    char face_type = (si == 0) ? '+' : '-';

    faces[si] = face;
    opposing_vertices[si] = opposing;

    // Create internal chord sub-edge
    int sub_idx = si + 2;  // '+' -> 2, '-' -> 3
    sub_edges[sub_idx] = mesh->make_edge(midpoint, opposing, false);

    // Find edges opposite to vertex0 and vertex1
    int idx_v1 = -1, idx_v0 = -1;
    for (int i = 0; i < 3; ++i) {
        if (face->vertices[i] == vertex1) idx_v1 = i;
        if (face->vertices[i] == vertex0) idx_v0 = i;
    }
    Edge* edge0 = face->edges[idx_v1];
    Edge* edge1 = face->edges[idx_v0];

    // Build mask split using the chord's half-plane
    Edge* chord = sub_edges[sub_idx];
    std::vector<bool> chord_mask(mesh->n_points, false);
    for (int i = 0; i < mesh->n_points; ++i) {
        double val = chord->normal[0] * mesh->get_x(i, 0) +
                     chord->normal[1] * mesh->get_x(i, 1);
        chord_mask[i] = (val >= chord->intercept);
    }

    std::vector<bool> mask0(mesh->n_points), mask1(mesh->n_points);
    std::string path0, path1;

    if (face_type == '+') {
        for (int i = 0; i < mesh->n_points; ++i) {
            mask0[i] = chord_mask[i] && face->mask[i];
            mask1[i] = !chord_mask[i] && face->mask[i];
        }
        path0 = face->path + '+';
        path1 = face->path + '-';
    } else {
        for (int i = 0; i < mesh->n_points; ++i) {
            mask0[i] = !chord_mask[i] && face->mask[i];
            mask1[i] = chord_mask[i] && face->mask[i];
        }
        path0 = face->path + '-';
        path1 = face->path + '+';
    }

    Face* face0 = mesh->make_face(edge0, sub_edges[0], sub_edges[sub_idx], mask0, false, path0);
    Face* face1 = mesh->make_face(edge1, sub_edges[sub_idx], sub_edges[1], mask1, false, path1);

    SubDivision sd;
    sd.e = chord;
    if (face_type == '+') {
        sd.plus_face = face0;
        sd.minus_face = face1;
    } else {
        sd.plus_face = face1;
        sd.minus_face = face0;
    }
    face->add_sub_division(this, sd);

    // Check aspect ratio
    double ar = std::max(face0->aspect_ratio(), face1->aspect_ratio());
    if (ar >= mesh->max_aspect_ratio) {
        disqualifying.insert(face);
        if (!midpoint->disqualified) {
            midpoint->disqualified = true;
            mesh->heap_pop(midpoint);
        }
    }

    midpoint->update_info();
}

void Edge::remove_face(Face* face) {
    if (faces[0] == face) faces[0] = nullptr;
    if (faces[1] == face) faces[1] = nullptr;

    disqualifying.erase(face);
    if (disqualifying.empty()) {
        midpoint->disqualified = false;
    }
}
