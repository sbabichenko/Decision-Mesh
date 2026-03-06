#pragma once
#include <string>

struct DecisionMesh;
struct Edge;
struct Face;

struct TreeNode {
    static int _seq;

    int _id;
    DecisionMesh* mesh = nullptr;
    TreeNode* parent = nullptr;
    Face* face = nullptr;

    // Split criterion (valid if !is_leaf())
    double split_normal[2] = {0, 0};
    double split_intercept = 0.0;
    bool has_split = false;

    TreeNode* p = nullptr;  // '+' child
    TreeNode* m = nullptr;  // '-' child

    TreeNode() : _id(-1) {}
    TreeNode(DecisionMesh* mesh, TreeNode* parent = nullptr, Face* face = nullptr);

    std::string sid() const;
    bool is_leaf() const { return !has_split; }
    int depth() const;
    std::string path() const;

    void split(Edge* edge, Face* plus, Face* minus);
    TreeNode* child_for(char sign) const;
};
