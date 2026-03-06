#include "vertex.h"
#include "edge.h"
#include "face.h"
#include "mesh.h"
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>

extern CascadeCounters g_counters;

int Vertex::_seq = 0;

Vertex::Vertex(DecisionMesh* mesh, double x, double y, bool active,
               Edge* parent_edge, bool real_vertex)
    : _id(Vertex::_seq++), x(x), y(y), active(active),
      real_vertex(real_vertex), mesh(mesh), parent_edge(parent_edge)
{
    if (real_vertex) {
        mesh->vertices.insert(this);
    }
    if (parent_edge) {
        height = (parent_edge->vertex0->height + parent_edge->vertex1->height) / 2.0;
        depth = std::max(parent_edge->vertex0->depth, parent_edge->vertex1->depth) + 1;
        // Register with parent edge endpoints for prior propagation
        if (real_vertex) {
            parent_edge->vertex0->prior_children.insert(this);
            parent_edge->vertex1->prior_children.insert(this);
        }
    } else {
        height = 0.0;
        depth = 0;
    }
}

std::string Vertex::sid() const {
    char buf[16];
    snprintf(buf, sizeof(buf), "V%03d", _id);
    return buf;
}

void Vertex::add_edge(Edge* e) {
    edges.insert(e);
    Vertex* other = e->other_vertex(this);
    if (other) neighbors.insert(other);
}

void Vertex::remove_edge(Edge* e) {
    edges.erase(e);
    Vertex* other = e->other_vertex(this);
    if (other) neighbors.erase(other);
}

void Vertex::activate() {
    if (active) return;
    g_counters.cascade_activations++;
    active = true;
    height = new_height;

    // Record detail coefficient before split (children may reference it)
    if (mesh->use_eb && parent_edge != nullptr) {
        delta_data = height - mu_lin();
    }

    parent_edge->split();

    std::set<Vertex*> to_update(affected_vertices);
    if (mesh->use_eb) {
        to_update.insert(prior_children.begin(), prior_children.end());
    }
    for (Vertex* v : to_update) {
        v->update_info();
    }
    loss_reduction = 0;
    mesh->heap_set(this, 0);
}

void Vertex::update_info() {
    g_counters.update_info_calls++;
    auto result = loc_regress();
    affected_vertices = result.affected;
    affected_points = result.n_points;

    double xTx = result.xTx;
    double xTr = result.xTr;
    double rTr = result.rTr;

    sigma_sq = (xTx > 0) ? (1.0 / xTx) : 1e300;

    if (mesh->use_eb) {
        compute_spike_slab(xTx, xTr);

        if (lambda_v > 0 && xTx > 0) {
            double mu_v = mu_lin();  // prior mean = linear interpolation
            double lv = lambda_v;
            double xTx_reg = xTx + lv;
            double xTr_reg = xTr + lv * mu_v;
            double beta_reg = xTr_reg / xTx_reg;

            double beta_orig = height;
            double orig_loss = beta_orig * beta_orig * xTx
                             - 2.0 * beta_orig * xTr + rTr
                             + lv * (beta_orig - mu_v) * (beta_orig - mu_v);
            double sse_post = rTr + lv * mu_v * mu_v - xTr_reg * xTr_reg / xTx_reg;

            new_height = beta_reg;
            loss_reduction = std::max(0.0, orig_loss - sse_post);
        } else {
            new_height = result.beta_opt;
            loss_reduction = result.loss_reduction;
            if (parent_edge != nullptr && xTx > 0) {
                delta_data = result.beta_opt - mu_lin();
            }
        }
    } else {
        new_height = result.beta_opt;
        loss_reduction = result.loss_reduction;
    }

    if (!disqualified) {
        mesh->heap_set(this, -loss_reduction);
    }
}

void Vertex::update_height() {
    height = new_height;
    if (parent_edge != nullptr) {
        delta_data = height - mu_lin();
    }
    std::set<Vertex*> to_update(affected_vertices);
    if (mesh->use_eb) {
        to_update.insert(prior_children.begin(), prior_children.end());
    }
    for (Vertex* v : to_update) {
        v->update_info();
    }
    loss_reduction = 0;
    mesh->heap_set(this, 0);
}

Vertex::FacesResult Vertex::get_faces() const {
    FacesResult result;
    for (Edge* e : edges) {
        for (int s = 0; s < 2; ++s) {
            Face* f = e->faces[s];
            if (f) {
                bool has_self = false;
                for (int i = 0; i < 3; ++i) {
                    if (f->vertices[i] == this) { has_self = true; break; }
                }
                if (has_self) {
                    result.faces.insert(f);
                }
            }
        }
    }
    for (Face* f : result.faces) {
        for (int i = 0; i < 3; ++i) {
            if (f->vertices[i] != this) {
                result.neighbors.insert(f->vertices[i]);
            }
        }
    }
    return result;
}

Vertex::FacesResult Vertex::get_sim_faces() const {
    FacesResult result;
    if (!parent_edge) return result;

    for (int s = 0; s < 2; ++s) {
        Face* face = parent_edge->faces[s];
        if (!face) continue;
        // Find which edge index in face corresponds to parent_edge
        int idx = -1;
        for (int i = 0; i < 3; ++i) {
            if (face->edges[i] == parent_edge) { idx = i; break; }
        }
        if (idx < 0) continue;
        if (face->sub_divisions[idx].plus_face)
            result.faces.insert(face->sub_divisions[idx].plus_face);
        if (face->sub_divisions[idx].minus_face)
            result.faces.insert(face->sub_divisions[idx].minus_face);
    }

    for (Face* f : result.faces) {
        for (int i = 0; i < 3; ++i) {
            if (f->vertices[i] != this) {
                result.neighbors.insert(f->vertices[i]);
            }
        }
    }
    return result;
}

std::set<Vertex*> Vertex::neighbors_after_split() const {
    std::set<Vertex*> nbrs;
    if (!parent_edge) return nbrs;
    nbrs.insert(const_cast<Vertex*>(this));
    nbrs.insert(parent_edge->vertex0);
    nbrs.insert(parent_edge->vertex1);
    if (parent_edge->opposing_vertices[0])
        nbrs.insert(parent_edge->opposing_vertices[0]);
    if (parent_edge->opposing_vertices[1])
        nbrs.insert(parent_edge->opposing_vertices[1]);
    return nbrs;
}

Vertex::LocRegressResult Vertex::loc_regress() {
    LocRegressResult res;
    res.beta_opt = height;
    res.loss_reduction = 0.0;
    res.n_points = 0;
    res.xTx = 0.0;
    res.xTr = 0.0;
    res.rTr = 0.0;

    FacesResult fr;
    if (active) {
        fr = get_faces();
    } else {
        fr = get_sim_faces();
    }
    res.affected = fr.neighbors;

    if (fr.faces.empty()) {
        return res;
    }

    double xTx = 0.0, xTr = 0.0, rTr = 0.0;
    int total_points = 0;

    for (Face* f : fr.faces) {
        int si = -1;
        for (int i = 0; i < 3; ++i) {
            if (f->vertices[i] == this) { si = i; break; }
        }
        if (si < 0 || f->n_covered == 0) continue;

        total_points += f->n_covered;

        double h[3];
        for (int i = 0; i < 3; ++i)
            h[i] = f->vertices[i]->height;

        xTx += f->S_ww[si][si];

        double xTr_face = f->S_wy[si];
        for (int j = 0; j < 3; ++j) {
            if (j == si) continue;
            xTr_face -= h[j] * f->S_ww[si][j];
        }
        xTr += xTr_face;

        double rTr_face = f->S_yy;
        for (int j = 0; j < 3; ++j) {
            if (j == si) continue;
            rTr_face -= 2.0 * h[j] * f->S_wy[j];
            for (int k = 0; k < 3; ++k) {
                if (k == si) continue;
                rTr_face += h[j] * h[k] * f->S_ww[j][k];
            }
        }
        rTr += rTr_face;
    }

    if (total_points == 0) return res;

    double beta_orig = height;
    double orig_loss = rTr - 2.0 * beta_orig * xTr + beta_orig * beta_orig * xTx;

    double beta_opt, sse_post;
    if (xTx > 0.0) {
        beta_opt = xTr / xTx;
        sse_post = rTr - (xTr * xTr) / xTx;
    } else {
        beta_opt = beta_orig;
        sse_post = rTr;
    }

    res.beta_opt = beta_opt;
    res.loss_reduction = orig_loss - sse_post;
    res.n_points = total_points;
    res.xTx = xTx;
    res.xTr = xTr;
    res.rTr = rTr;
    return res;
}

// --- Empirical Bayes helpers ---

double Vertex::mu_lin() const {
    if (parent_edge == nullptr) return 0.0;
    return (parent_edge->vertex0->height + parent_edge->vertex1->height) / 2.0;
}

void Vertex::compute_spike_slab(double xTx, double xTr) {
    // Sets s_v, lambda_v, p_v using wavelet spike-and-slab posterior.
    // Prior: delta ~ (1 - pi_d) * delta(0) + pi_d * N(0, tau_sq_d)
    // Data:  delta_hat | delta ~ N(delta, sigma_sq)

    static constexpr double EPS_S = 1e-6;  // floor for s_v to avoid lambda_v -> inf

    if (parent_edge == nullptr || xTx <= 0) {
        s_v = 0.0; p_v = 0.0; lambda_v = 0.0;
        return;
    }

    auto tau_it = mesh->tau_sq.find(depth);
    auto pi_it = mesh->pi_d.find(depth);
    if (tau_it == mesh->tau_sq.end() || pi_it == mesh->pi_d.end()) {
        s_v = 0.0; p_v = 0.0; lambda_v = 0.0;
        return;
    }

    double tau2 = tau_it->second;
    double pi = pi_it->second;
    double sig2 = 1.0 / xTx;  // sigma_sq_v

    // Data-only detail coefficient
    double beta_data = xTr / xTx;
    double ml = mu_lin();
    double delta_hat = beta_data - ml;
    delta_data = delta_hat;

    if (tau2 <= 0) {
        // All spike: full shrinkage to linear interpolation
        s_v = EPS_S;
        p_v = 0.0;
        lambda_v = xTx * (1.0 - EPS_S) / EPS_S;
        return;
    }

    // Bayes factor: slab likelihood / spike likelihood
    // BF = sqrt(sig2 / (sig2 + tau2)) * exp(delta_hat^2 * tau2 / (2 * sig2 * (sig2 + tau2)))
    double ratio = sig2 / (sig2 + tau2);
    double exponent = delta_hat * delta_hat * tau2 / (2.0 * sig2 * (sig2 + tau2));

    // Clamp exponent to avoid overflow
    double log_bf = 0.5 * std::log(ratio) + exponent;
    // BF = exp(log_bf)

    // Posterior inclusion: p_v = 1 / (1 + ((1-pi)/pi) * exp(-log_bf))
    double log_odds_prior = std::log(pi / (1.0 - pi));
    double log_odds_post = log_odds_prior + log_bf;

    if (log_odds_post > 30.0) {
        p_v = 1.0;
    } else if (log_odds_post < -30.0) {
        p_v = 0.0;
    } else {
        p_v = 1.0 / (1.0 + std::exp(-log_odds_post));
    }

    // Slab shrinkage component
    double slab_shrink = tau2 / (tau2 + sig2);

    // Combined shrinkage
    s_v = std::max(EPS_S, p_v * slab_shrink);

    // Effective regularization: lambda_v = xTx * (1 - s_v) / s_v
    lambda_v = xTx * (1.0 - s_v) / s_v;
}
