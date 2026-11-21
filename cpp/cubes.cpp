// SpOC3 Programmable Cubes: fast simulator + greedy solver (C++ port of cubesim.py / greedy.py).
//
// Build:  g++ -O2 -march=native -std=c++17 -o cubes cpp/cubes.cpp
// Input:  a text file written by export_problem.py
// Usage:  ./cubes solve   <problem.txt> <seed> <urgent> <out.txt> [time_limit_s]
//         ./cubes simulate <problem.txt> <commands.txt>      (prints correct cubes, legal moves)
// Output: one "cube move" pair per line.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

// ---------------------------------------------------------------- move tables (CubeMoveset.py)
static const vector<vector<int>> PAT_OCC = {{7, 8}, {6, 11}, {15, 16}, {12, 17}, {6, 7}, {11, 15}, {16, 17}, {8, 12},
                                            {16},   {7},     {11},     {12},     {12},   {11},     {16},     {7}};
static const vector<vector<int>> PAT_EMPTY = {
    {12, 16, 17}, {7, 8, 12}, {6, 7, 11}, {11, 15, 16}, {11, 15, 16}, {12, 16, 17}, {7, 8, 12}, {6, 7, 11},
    {14, 15, 10, 11, 6, 7}, {8, 9, 12, 13, 16, 17}, {12, 6, 7, 8, 1, 2}, {11, 15, 16, 17, 21, 22},
    {11, 6, 7, 8, 2, 3}, {12, 15, 16, 17, 20, 21}, {17, 18, 12, 13, 7, 8}, {5, 6, 10, 11, 15, 16}};
static const int MAPPING[2][8] = {{0, 1, 2, 3, 8, 9, 10, 11}, {4, 5, 6, 7, 12, 13, 14, 15}};
static const int DIFFS[16][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {-1, 0}, {0, -1}, {1, 0},  {0, 1},
                                 {-1, -1}, {1, 1}, {-1, 1}, {1, -1}, {1, 1}, {-1, -1}, {1, -1}, {-1, 1}};

struct V3 { int x, y, z; };
static V3 plane(int ax, int j, int i) {
    if (ax == 0) return {j, i, 0};
    if (ax == 1) return {0, j, i};
    return {j, 0, i};
}

// Grid of cells, linear index. Offsets become index deltas.
static int GX, GY, GZ, OX, OY, OZ;  // size and origin shift
static inline int IDX(int x, int y, int z) { return ((x + OX) * GY + (y + OY)) * GZ + (z + OZ); }
static inline int DELTA(V3 v) { return (v.x * GY + v.y) * GZ + v.z; }

struct Opt { vector<int> occ, emp; int disp; };
static vector<Opt> MOVES[6];
static int NB6[6];
static int MARGIN = 3;  // cells with any coordinate within MARGIN of the border are off limits

static void build_tables() {
    for (int m = 0; m < 6; m++) {
        int ax = m / 2, d = m % 2;
        vector<V3> grid;
        for (int i = 2; i >= -2; i--)
            for (int j = -2; j <= 2; j++)
                if (i != 0 || j != 0) grid.push_back(plane(ax, j, i));
        MOVES[m].clear();
        for (int k = 0; k < 8; k++) {
            int p = MAPPING[d][k];
            Opt o;
            for (int c : PAT_OCC[p]) o.occ.push_back(DELTA(grid[c]));
            for (int c : PAT_EMPTY[p]) o.emp.push_back(DELTA(grid[c]));
            o.disp = DELTA(plane(ax, DIFFS[p][0], DIFFS[p][1]));
            MOVES[m].push_back(o);
        }
    }
    V3 n6[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int i = 0; i < 6; i++) NB6[i] = DELTA(n6[i]);
}

// ---------------------------------------------------------------- problem + state
struct Problem {
    int n, ntypes, max_cmds;
    double offset;
    vector<int> init_pos, init_type;   // cell index per cube
    vector<int> target_type;           // per cell: -1 not target, else type
    vector<int> target_cells;          // list of target cells (file order)
    vector<char> inside;               // cell is far enough from the border
};

static Problem P;

struct State {
    vector<int> pos;   // cube -> cell
    vector<int> occ;   // cell -> cube or -1
    int correct = 0;
    uint64_t h = 0;
};

static vector<uint64_t> ZOB;

static inline bool done_cube(const State& s, int c) { return P.target_type[s.pos[c]] == P.init_type[c]; }

static State init_state() {
    State s;
    s.pos = P.init_pos;
    s.occ.assign(GX * GY * GZ, -1);
    for (int c = 0; c < P.n; c++) {
        s.occ[s.pos[c]] = c;
        s.h ^= ZOB[s.pos[c]];
        s.correct += done_cube(s, c);
    }
    return s;
}

static inline bool occupied(const State& s, int q, int ignore) { return q != ignore && s.occ[q] != -1; }

// Where a cube at p lands with move m (cell `ignore` treated as empty), or -1.
static inline int land(const State& s, int p, int m, int ignore) {
    for (const Opt& o : MOVES[m]) {
        bool ok = true;
        for (int d : o.occ)
            if (!occupied(s, p + d, ignore)) { ok = false; break; }
        if (!ok) continue;
        for (int d : o.emp)
            if (occupied(s, p + d, ignore)) { ok = false; break; }
        if (ok) {
            int q = p + o.disp;
            return P.inside[q] ? q : -1;
        }
    }
    return -1;
}

static int count_nb(const State& s, int p) {
    int k = 0;
    for (int d : NB6) k += s.occ[p + d] != -1;
    return k;
}

static void move_cube(State& s, int c, int q) {
    int p = s.pos[c];
    s.correct -= done_cube(s, c);
    s.occ[p] = -1;
    s.h ^= ZOB[p];
    s.pos[c] = q;
    s.occ[q] = c;
    s.h ^= ZOB[q];
    s.correct += done_cube(s, c);
}

// Official connectivity rule: removing cube c keeps the ensemble connected.
static vector<int> stampv;
static int stamp_id = 0;
static bool is_removable(const State& s, int c) {
    int p = s.pos[c];
    int nbs[6], k = 0;
    for (int d : NB6)
        if (s.occ[p + d] != -1) nbs[k++] = s.occ[p + d];
    if (k <= 1) return true;
    if ((int)stampv.size() < P.n) stampv.assign(P.n, 0);
    stamp_id++;
    vector<int> q;
    q.reserve(64);
    q.push_back(nbs[0]);
    stampv[nbs[0]] = stamp_id;
    stampv[c] = stamp_id;
    int found = 1;
    for (size_t i = 0; i < q.size(); i++) {
        int u = q[i];
        for (int d : NB6) {
            int v = s.occ[s.pos[u] + d];
            if (v == -1 || stampv[v] == stamp_id) continue;
            stampv[v] = stamp_id;
            for (int j = 0; j < k; j++)
                if (nbs[j] == v && ++found == k) return true;
            q.push_back(v);
        }
    }
    return false;
}

static bool try_move(State& s, int c, int m) {
    if (!is_removable(s, c)) return false;
    int q = land(s, s.pos[c], m, -1);
    if (q < 0) return false;
    move_cube(s, c, q);
    return true;
}

// Articulation points (iterative Tarjan) of the face-adjacency graph.
static vector<char> articulation(const State& s) {
    int n = P.n;
    vector<int> disc(n, -1), low(n, 0), par(n, -1), it(n, 0);
    vector<char> ap(n, 0);
    int t = 0;
    for (int root = 0; root < n; root++) {
        if (disc[root] != -1) continue;
        int children = 0;
        vector<int> st = {root};
        disc[root] = low[root] = t++;
        while (!st.empty()) {
            int u = st.back();
            if (it[u] < 6) {
                int v = s.occ[s.pos[u] + NB6[it[u]++]];
                if (v == -1) continue;
                if (disc[v] == -1) {
                    par[v] = u;
                    disc[v] = low[v] = t++;
                    if (u == root) children++;
                    st.push_back(v);
                } else if (v != par[u]) {
                    low[u] = min(low[u], disc[v]);
                }
            } else {
                st.pop_back();
                if (!st.empty()) {
                    int w = st.back();
                    low[w] = min(low[w], low[u]);
                    if (w != root && low[u] >= disc[w]) ap[w] = 1;
                }
            }
        }
        if (children > 1) ap[root] = 1;
    }
    return ap;
}

// ---------------------------------------------------------------- search
static vector<int> par_cell, par_move, vis;
static int vis_id = 0;

static void ensure_bufs() {
    if (vis.empty()) {
        int G = GX * GY * GZ;
        vis.assign(G, 0);
        par_cell.assign(G, -1);
        par_move.assign(G, -1);
    }
}

typedef pair<int, vector<int>> Found;  // cube, moves

static int forward_end(const State& s, int start, const vector<int>& moves) {
    int p = start;
    for (int m : moves) {
        p = land(s, p, m, start);
        if (p < 0) return -1;
    }
    return p;
}

struct PairHash { size_t operator()(const pair<int, int>& a) const { return (uint64_t)a.first * 1000003ULL ^ a.second; } };
typedef unordered_set<pair<int, int>, PairHash> Banned;

// Can a cube still pivot into empty cell e (from an empty cell or from a cube that is not yet placed)?
static bool enterable(const State& s, int e) {
    for (int m = 0; m < 6; m++)
        for (const Opt& o : MOVES[m]) {
            int r = e - o.disp;
            if (!P.inside[r]) continue;
            int c = s.occ[r];
            if (c != -1 && done_cube(s, c)) continue;
            if (land(s, r, m, r) == e) return true;
        }
    return false;
}

static vector<int> NEAR;  // offsets within Chebyshev distance 2
// Would moving the cube at r to q seal off a nearby empty target cell that is enterable now?
static bool seals(State& s, int r, int q) {
    if (NEAR.empty())
        for (int dx = -2; dx <= 2; dx++)
            for (int dy = -2; dy <= 2; dy++)
                for (int dz = -2; dz <= 2; dz++)
                    if (dx || dy || dz) NEAR.push_back(DELTA({dx, dy, dz}));
    int cand[124], k = 0;
    for (int d : NEAR) {
        int e = q + d;
        if (e == r || P.target_type[e] < 0 || s.occ[e] != -1) continue;
        if (enterable(s, e)) cand[k++] = e;
    }
    if (!k) return false;
    int c = s.occ[r];
    move_cube(s, c, q);
    bool bad = false;
    for (int i = 0; i < k && !bad; i++) bad = !enterable(s, cand[i]);
    move_cube(s, c, r);
    return bad;
}

static bool SEAL_CHECK = false;
static Banned SEALBAN;

// Reverse BFS from all goals of each type; movers[c] says cube c may be moved.
static bool find_move(const State& s, const vector<char>& movers, const vector<vector<int>>& goals_of_type,
                      Banned& banned, int max_nodes, Found& best) {
    ensure_bufs();
    bool have = false;
    for (int t = 0; t < P.ntypes; t++) {
        const vector<int>& goals = goals_of_type[t];
        if (goals.empty()) continue;
        bool any = false;
        for (int c = 0; c < P.n && !any; c++) any = movers[c] && P.init_type[c] == t;
        if (!any) continue;
        vis_id++;
        deque<int> dq;
        for (int g : goals) { vis[g] = vis_id; par_cell[g] = -1; dq.push_back(g); }
        int nodes = 0;
        bool stop = false;
        while (!dq.empty() && nodes < max_nodes && !stop) {
            int p = dq.front();
            dq.pop_front();
            nodes++;
            for (int m = 0; m < 6 && !stop; m++) {
                for (const Opt& o : MOVES[m]) {
                    int r = p - o.disp;
                    if (!P.inside[r] || vis[r] == vis_id) continue;
                    if (land(s, r, m, r) != p) continue;
                    int c = s.occ[r];
                    if (c != -1) {
                        if (movers[c] && P.init_type[c] == t && !banned.count({c, p}) &&
                            !(SEAL_CHECK && SEALBAN.count({c, p}))) {
                            vector<int> moves = {m};
                            int q = p;
                            while (par_cell[q] != -1) { moves.push_back(par_move[q]); q = par_cell[q]; }
                            int end = forward_end(s, r, moves);
                            if (end == q && SEAL_CHECK && seals(const_cast<State&>(s), r, q)) {
                                SEALBAN.insert({c, p});
                                continue;
                            }
                            if (end == q) {
                                if (!have || moves.size() < best.second.size()) { best = {c, moves}; have = true; }
                                stop = true;
                                break;
                            }
                            banned.insert({c, p});
                        }
                        continue;
                    }
                    vis[r] = vis_id;
                    par_cell[r] = p;
                    par_move[r] = m;
                    dq.push_back(r);
                }
            }
        }
    }
    return have;
}

// Forward BFS for the cube at start (lifted out); returns moves to the first cell with accept(cell).
template <class F>
static bool forward_bfs(const State& s, int start, F accept, int max_nodes, vector<int>& out) {
    ensure_bufs();
    vis_id++;
    deque<int> dq = {start};
    vis[start] = vis_id;
    par_cell[start] = -1;
    int nodes = 1;
    while (!dq.empty() && nodes < max_nodes) {
        int p = dq.front();
        dq.pop_front();
        for (int m = 0; m < 6; m++) {
            int q = land(s, p, m, start);
            if (q < 0 || vis[q] == vis_id || (s.occ[q] != -1 && q != start)) continue;
            vis[q] = vis_id;
            par_cell[q] = p;
            par_move[q] = m;
            nodes++;
            if (accept(q)) {
                out.clear();
                int r = q;
                while (par_cell[r] != -1) { out.push_back(par_move[r]); r = par_cell[r]; }
                reverse(out.begin(), out.end());
                return true;
            }
            dq.push_back(q);
        }
    }
    return false;
}

static double fitness_from(int correct, int ncmds) {
    double F = 0.9 * correct / P.n + 0.1 * (1.0 - (double)ncmds / P.max_cmds);
    return -(F - P.offset) / (1 - P.offset);
}

// ---------------------------------------------------------------- solver
struct Params {
    int urgent = 3, max_repairs = 200, max_shakes = 300, max_nodes = 200000;
    double time_limit = 1e18;
    uint64_t seed = 0;
    bool seal = true;
};

static vector<pair<int, int>> solve(const Params& prm) {
    State s = init_state();
    vector<pair<int, int>> cmds;
    mt19937_64 rng(prm.seed);
    auto t0 = chrono::steady_clock::now();
    unordered_set<uint64_t> seen;
    unordered_map<int, int> doors;  // door cell -> pocket cell
    int repairs = 0, shakes = 0;
    pair<int, int> best = {s.correct, 0};
    Banned banned;
    vector<vector<int>> goals(P.ntypes), tier(P.ntypes);

    while (s.correct < P.n) {
        double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        if (el > prm.time_limit) break;
        vector<char> ap = articulation(s);
        vector<char> movable(P.n, 0), squat(P.n, 0);
        bool any_squat = false;
        for (int c = 0; c < P.n; c++)
            if (!done_cube(s, c) && !ap[c]) {
                movable[c] = 1;
                if (P.target_type[s.pos[c]] >= 0) { squat[c] = 1; any_squat = true; }
            }
        for (auto it = doors.begin(); it != doors.end();)
            it = (s.occ[it->second] != -1) ? doors.erase(it) : next(it);
        for (auto& g : goals) g.clear();
        for (int cell : P.target_cells)
            if (s.occ[cell] == -1 && !doors.count(cell)) goals[P.target_type[cell]].push_back(cell);
        for (auto& g : goals) shuffle(g.begin(), g.end(), rng);

        Found f;
        bool have = false;
        if (any_squat) have = find_move(s, squat, goals, banned, prm.max_nodes, f);
        if (!have && any_squat) {
            vector<int> mv;
            for (int c = 0; c < P.n; c++) {
                if (!squat[c]) continue;
                if (forward_bfs(s, s.pos[c], [&](int q) { return P.target_type[q] < 0; }, 50000, mv))
                    if (!have || mv.size() < f.second.size()) { f = {c, mv}; have = true; }
            }
        }
        SEALBAN.clear();
        for (int pass = (prm.seal ? 0 : 1); pass < 2 && !have; pass++) {
        SEAL_CHECK = (pass == 0);
        if (!have && prm.urgent > 0) {
            for (int u = 6; u >= prm.urgent && !have; u--) {
                for (int t = 0; t < P.ntypes; t++) {
                    tier[t].clear();
                    for (int g : goals[t])
                        if (count_nb(s, g) >= u) tier[t].push_back(g);
                }
                have = find_move(s, movable, tier, banned, prm.max_nodes, f);
            }
        }
        if (!have) have = find_move(s, movable, goals, banned, prm.max_nodes, f);
        }
        SEAL_CHECK = false;
        // repair: a correct cube moves into a walled-in cell of its type if it leaves an easier hole
        if (!have && repairs < prm.max_repairs) {
            vector<char> donem(P.n, 0);
            for (int c = 0; c < P.n; c++) donem[c] = done_cube(s, c) && !ap[c];
            for (int t = 0; t < P.ntypes; t++) {
                tier[t].clear();
                for (int g : goals[t])
                    if (count_nb(s, g) >= 3) tier[t].push_back(g);
            }
            Banned b2;
            for (int k = 0; k < 30; k++) {
                Found g;
                if (!find_move(s, donem, tier, b2, prm.max_nodes, g)) break;
                int start = s.pos[g.first];
                int end = forward_end(s, start, g.second);
                int goal_n = count_nb(s, end);
                int dd = abs(start - end);
                bool adj = false;
                for (int d : NB6) adj |= (start + d == end);
                (void)dd;
                int left_n = count_nb(s, start) + (adj ? 1 : 0);
                uint64_t key = s.h ^ ZOB[start] ^ ZOB[end];
                if (left_n < goal_n && !seen.count(key)) {
                    seen.insert(key);
                    f = g;
                    have = true;
                    repairs++;
                    break;
                }
                b2.insert({g.first, land(s, start, g.second[0], start)});
            }
        }
        // open a door: park a correct neighbour of an unreachable empty target
        if (!have && repairs < prm.max_repairs) {
            vector<int> all;
            for (auto& g : goals) all.insert(all.end(), g.begin(), g.end());
            stable_sort(all.begin(), all.end(), [&](int a, int b) { return count_nb(s, a) > count_nb(s, b); });
            for (int g : all) {
                for (int d : NB6) {
                    int c = s.occ[g + d];
                    if (c == -1 || ap[c] || !done_cube(s, c)) continue;
                    int start = s.pos[c];
                    vector<int> mv;
                    if (!forward_bfs(s, start, [&](int q) { return P.target_type[q] < 0 && q != start; }, 50000, mv))
                        continue;
                    int end = forward_end(s, start, mv);
                    uint64_t key = s.h ^ ZOB[start] ^ ZOB[end];
                    if (seen.count(key)) continue;
                    seen.insert(key);
                    doors[start] = g;
                    f = {c, mv};
                    have = true;
                    repairs++;
                    break;
                }
                if (have) break;
            }
        }
        // shake: move a leftover cube to a random reachable non-target cell
        if (!have && shakes < prm.max_shakes) {
            vector<int> loose;
            for (int c = 0; c < P.n; c++)
                if (!done_cube(s, c) && !ap[c]) loose.push_back(c);
            shuffle(loose.begin(), loose.end(), rng);
            for (size_t i = 0; i < loose.size() && i < 20 && !have; i++) {
                int c = loose[i], start = s.pos[c];
                // collect reachable cells
                vector<int> cells;
                vector<int> dummy;
                forward_bfs(s, start, [&](int q) { if (P.target_type[q] < 0) cells.push_back(q); return false; }, 5000, dummy);
                if (cells.empty()) continue;
                int q = cells[rng() % cells.size()];
                uint64_t key = s.h ^ ZOB[start] ^ ZOB[q];
                if (seen.count(key)) continue;
                seen.insert(key);
                vector<int> mv;
                forward_bfs(s, start, [&](int x) { return x == q; }, 1 << 30, mv);
                f = {c, mv};
                have = true;
                shakes++;
            }
        }
        if (!have) {
            if (getenv("CUBES_DEBUG"))
                fprintf(stderr, "stuck: correct %d repairs %d shakes %d doors %zu seen %zu\n", s.correct, repairs, shakes,
                        doors.size(), seen.size());
            break;
        }
        for (int m : f.second) {
            int q = land(s, s.pos[f.first], m, -1);
            if (q < 0) { fprintf(stderr, "internal error: path illegal\n"); exit(2); }
            move_cube(s, f.first, q);
            cmds.push_back({f.first, m});
        }
        banned.clear();
        if (fitness_from(s.correct, cmds.size()) < fitness_from(best.first, best.second))
            best = {s.correct, (int)cmds.size()};
    }
    cmds.resize(best.second);
    return cmds;
}

// ---------------------------------------------------------------- I/O
static void load_problem(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) { perror(path); exit(1); }
    int n, nt, mc, nt_cells;
    double off;
    if (fscanf(fp, "%d %d %d %lf %d", &n, &nt, &mc, &off, &nt_cells) != 5) exit(1);
    vector<array<int, 4>> ini(n), tgt(nt_cells);
    int mn[3] = {1 << 30, 1 << 30, 1 << 30}, mx[3] = {-(1 << 30), -(1 << 30), -(1 << 30)};
    auto upd = [&](array<int, 4>& a) { for (int k = 0; k < 3; k++) { mn[k] = min(mn[k], a[k]); mx[k] = max(mx[k], a[k]); } };
    for (auto& a : ini) { if (fscanf(fp, "%d %d %d %d", &a[0], &a[1], &a[2], &a[3]) != 4) exit(1); upd(a); }
    for (auto& a : tgt) { if (fscanf(fp, "%d %d %d %d", &a[0], &a[1], &a[2], &a[3]) != 4) exit(1); upd(a); }
    fclose(fp);
    const int PAD = 14;
    OX = -mn[0] + PAD; OY = -mn[1] + PAD; OZ = -mn[2] + PAD;
    GX = mx[0] - mn[0] + 1 + 2 * PAD; GY = mx[1] - mn[1] + 1 + 2 * PAD; GZ = mx[2] - mn[2] + 1 + 2 * PAD;
    build_tables();
    P.n = n; P.ntypes = nt; P.max_cmds = mc; P.offset = off;
    int G = GX * GY * GZ;
    P.target_type.assign(G, -1);
    P.inside.assign(G, 0);
    for (int x = 0; x < GX; x++)
        for (int y = 0; y < GY; y++)
            for (int z = 0; z < GZ; z++)
                if (x >= MARGIN && y >= MARGIN && z >= MARGIN && x < GX - MARGIN && y < GY - MARGIN && z < GZ - MARGIN)
                    P.inside[(x * GY + y) * GZ + z] = 1;
    for (auto& a : ini) { P.init_pos.push_back(IDX(a[0], a[1], a[2])); P.init_type.push_back(a[3]); }
    for (auto& a : tgt) { int c = IDX(a[0], a[1], a[2]); P.target_type[c] = a[3]; P.target_cells.push_back(c); }
    mt19937_64 zr(12345);
    ZOB.resize(G);
    for (auto& z : ZOB) z = zr();
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "see header for usage\n"); return 1; }
    string mode = argv[1];
    load_problem(argv[2]);
    if (mode == "simulate") {
        State s = init_state();
        FILE* fp = fopen(argv[3], "r");
        int c, m, n = 0, legal = 0;
        while (fscanf(fp, "%d %d", &c, &m) == 2) { n++; legal += try_move(s, c, m); }
        printf("commands %d legal %d correct %d fitness %.10f\n", n, legal, s.correct, fitness_from(s.correct, n));
        return 0;
    }
    if (mode == "solve") {
        Params prm;
        prm.seed = strtoull(argv[3], 0, 10);
        prm.urgent = atoi(argv[4]);
        if (argc > 6) prm.time_limit = atof(argv[6]);
        if (argc > 7) prm.max_shakes = atoi(argv[7]);
        if (argc > 8) prm.max_repairs = atoi(argv[8]);
        if (argc > 9) prm.seal = atoi(argv[9]) != 0;
        auto t0 = chrono::steady_clock::now();
        auto cmds = solve(prm);
        State s = init_state();
        for (auto& cm : cmds)
            if (!try_move(s, cm.first, cm.second)) { fprintf(stderr, "illegal move in output\n"); return 2; }
        FILE* fp = fopen(argv[5], "w");
        for (auto& cm : cmds) fprintf(fp, "%d %d\n", cm.first, cm.second);
        fclose(fp);
        double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        printf("seed %llu urgent %d correct %d/%d moves %zu fitness %.6f time %.1fs\n", (unsigned long long)prm.seed,
               prm.urgent, s.correct, P.n, cmds.size(), fitness_from(s.correct, cmds.size()), el);
        return 0;
    }
    return 1;
}
