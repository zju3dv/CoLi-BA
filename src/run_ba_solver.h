
#pragma once

#include "base/ba_graph.h"

constexpr int _max_iter_ = 5;

void RunCeres(coli::BAGraph&ba_graph);

void RunCoLi(coli::BAGraph &ba_graph);

void RunG2O(coli::BAGraph &ba_graph);

// void RunSAM(coli::BAGraph ba_graph);
