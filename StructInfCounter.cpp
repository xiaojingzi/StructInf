// counter.cpp
//   Implementation of the structure-influence mining algorithm in C++11.
//   Since C++11 features are used, please use gcc 4.8+ or visual studio 2013+ to compile:)
//
// Copyright 2016 KEG, Tsinghua.
// originally created by Yuanyi Zhong

#define NDEBUG

// For some datasets, a (user,action) may only appear once. In that case,
// uncomment the the next line.
//#define ASSUME_ONE_ACTION_PER_USER

// Assume directed graph by default.
//#define UNDIRECTED_GRAPH

#include <cstdio>
#include <ctime>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
//#include <map>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <deque>
//#include <list>
//#include <functional>
#include <algorithm>
//#include <iterator>
#include <random>
#include <cassert>
#include <memory>
using namespace std;


///////////////////////////////////////////////////////////
// code for pattern matching

// Consider 20 different patterns consisting of 2,3,or 4 nodes.
// 'PATTERN_BAD' refers to the 13rd(from 0) pattern, which is the same as 8th pattern
// when represented by in-/out-degrees list. Thus need special case consideration.
#define PATTERN_COUNT	20
#define PATTERN_BAD		13

//
// lists of in-/out-degrees of each node, sorted in increasing order, e.g.
//   (in-deg-0, out-deg-0,  in-deg-1, out-deg-1,  ...)
//
//static int code_1_node[] = { 0 };
static const int code_2_nodes[][4] =
/* 00 */{ 0, 1, 1, 0 };
static const int code_3_nodes[][6] = {
/* 01 */{ 0, 1, 1, 0, 1, 1 },
/* 02 */{ 0, 1, 0, 1, 2, 0 },
/* 03 */{ 0, 2, 1, 1, 2, 0 }
};
static const int code_4_nodes[][8] = {
/* 04 */{ 0, 1, 1, 0, 1, 1, 1, 1 },
/* 05 */{ 0, 1, 0, 1, 1, 1, 2, 0 },
/* 06 */{ 0, 1, 0, 1, 1, 0, 2, 1 },
/* 07 */{ 0, 1, 0, 1, 0, 1, 3, 0 },
/* 08 */{ 0, 2, 1, 1, 1, 1, 2, 0 }, // #

/* 09 */{ 0, 1, 1, 1, 1, 2, 2, 0 },
/* 10 */{ 0, 2, 1, 0, 1, 1, 2, 1 },
/* 11 */{ 0, 1, 0, 2, 2, 0, 2, 1 },
/* 12 */{ 0, 1, 0, 2, 1, 1, 3, 0 },
/* 13 *///{ 0, 2, 1, 1, 2, 0, 1, 1 },	// conflict with #, PATTERN_BAD
	
/* 14 */{ 0, 2, 1, 1, 1, 2, 3, 0 },
/* 15 */{ 0, 3, 1, 1, 2, 0, 2, 1 },
/* 16 */{ 0, 2, 1, 2, 2, 0, 2, 1 },
/* 17 */{ 0, 3, 1, 1, 1, 1, 3, 0 },
/* 18 */{ 0, 2, 0, 2, 2, 1, 3, 0 },
	
/* 19 */{ 0, 3, 1, 2, 2, 1, 3, 0 }
};
// a map from hashed degree-list representation to pattern index
unordered_map<int, int> code_to_index;
int code_bad;

// This function hashes the degree lists to more compact representations.
void calc_pattern_codes() {
	int num = 0;
	for (auto &a : code_2_nodes) {
		int code = 0;
		for (auto &p : a) {
			code <<= 2;
			code += p;
		}
		code_to_index[code] = num++;
	}
	for (auto &a : code_3_nodes) {
		int code = 0;
		for (auto &p : a) {
			code <<= 2;
			code += p;
		}
		code_to_index[code] = num++;
	}
	for (auto &a : code_4_nodes) {
		int code = 0;
		for (auto &p : a) {
			code <<= 2;
			code += p;
		}
		if (num == 8)
			code_bad = code;
		else if (num == PATTERN_BAD)
			num++;
		code_to_index[code] = num++;
	}
}
///////////////////////////////////////////////////////////
// code for sampling methods

// 'X' holds the raw counts for each active/positive influence pattern, and
// 'Y' holds those for each inactive/negative pattern.
long long X[PATTERN_COUNT], Y[PATTERN_COUNT];
// estimated real pattern counts, i.e. estX = X/Sampling_probability
double estX[PATTERN_COUNT], estY[PATTERN_COUNT];

// Edge and vertex counts of each pattern.
static const int N_EDGE[PATTERN_COUNT] = { 1,2,2,3,3, 3,3,3,4,4, 4,4,4,4,5, 5,5,5,5,6 };
static const int N_VERT[PATTERN_COUNT] = { 2,3,3,3,4, 4,4,4,4,4, 4,4,4,4,4, 4,4,4,4,4 };

// "scaling factor":
// S[1][i] = 1 / sampling probability of positive pattern_i
// S[0][i] = 1 / sampling probability of negative pattern_i
static double S[2][PATTERN_COUNT];
void reset_S() {
	for (int i = 0; i < 2; ++i)
		for (auto &x : S[i]) x = 1;
}

// define bernoulli random distributions for sampling
std::random_device rd;
std::mt19937 gen(rd());

bernoulli_distribution bernoulli_edge(1);
bernoulli_distribution bernoulli_vert[2] = {bernoulli_distribution(1), bernoulli_distribution(1)};
//bernoulli_distribution bernoulli_neg(1);

//
// 'set_edge_prob' & 'set_vert_prob' set the sampling probabilities
// for edges and vertives respectively, i.e the q, px, py parameters metioned in the paper.
//
void set_edge_prob(double q) {
	for (int i = 0; i < PATTERN_COUNT; ++i) {
		auto v = pow(1 / q, N_EDGE[i]);
		S[0][i] *= v; S[1][i] *= v;
	}
	bernoulli_edge.param(bernoulli_distribution::param_type(q));
}
void set_vert_prob(double px, double py) {
	// Sampling probabilities in each level of the enumeration tree can
	// actually be different, in order to achieve a smaller variance of the result.
	//for (int i = 0; i < 4; ++i)
	//	bernoulli_vert[i].param(bernoulli_distribution::param_type(p));
	bernoulli_vert[1].param(bernoulli_distribution::param_type(px));
	bernoulli_vert[0].param(bernoulli_distribution::param_type(py));
	double P[5] = { 1, px, px*px, px*px*px, px*px*px*px };
	double Q[5] = { 1, py, py*py, py*py*py, py*py*py*py };
	//double P[5] = { 1, pow(p, 0.5), pow(p, 0.5+1), pow(p, 0.5+1+1.5), pow(p, 0.5+1+1.5+2) };
	for (int i = 0; i < PATTERN_COUNT; ++i) {
		S[0][i] /= Q[N_VERT[i]];
		S[1][i] /= P[N_VERT[i]];
	}
}

/*void set_neg_p(double p) {
	for (int i = 0; i < PATTERN_COUNT; ++i)
		S[0][i] *= 1 / p;
	bernoulli_neg.param(bernoulli_distribution::param_type(p));
}*/

/////////////////////////////////////////////////////////////////
// pattern counter

// class for a node in action-diffusion-graph, i.e., a log entry/propagation node.
// to be more specific, contains:
//    (uid, aid, time) - (user, action, time) tuple,
//    gid - global unique (increasing) identifier
//    in - incoming links of the node
struct prop_node {
	//typedef shared_ptr<prop_node>	prop_node_ptr;
	typedef prop_node *prop_node_ptr;
	typedef unordered_set<prop_node_ptr>	unordered_subgraph_t;

	int uid;
	int aid;
	int time;

	int gid;

	unordered_subgraph_t in;	// pointers to prop_nodes
	//unordered_subgraph_t out;
};
using prop_node_ptr = prop_node::prop_node_ptr;

// 'ordered_subgraph_t' is the container for subgraph nodes ordered by 'gid'
struct ordered_subgraph_less {
	bool operator()(const prop_node_ptr &a, const prop_node_ptr &b) const { 
		//return a->time > b->time;	// times can be the same, thus not good.
		return a->gid > b->gid;
	}
};
//typedef multiset<prop_node_ptr, ordered_subgraph_less> ordered_subgraph_t;
typedef set<prop_node_ptr, ordered_subgraph_less> ordered_subgraph_t;

template<class A, class B>
inline bool contains(A &a, B &b) {
	return a.find(b) != a.end();
}

//
// Function: countPattern
//    It induces the subgraph from a node set 'sub', tries to match the
//    subgraph to a predefined influence pattern, and increases the count
//    of matched pattern.
//
// Templete arguments:
//    b_pos - whether it's counting positive pattern or not;
//
// Inputs:
//    sub - nodes (ordered by gid) of a potential pattern.
//
template<bool b_pos>
void countPattern(ordered_subgraph_t &sub) {
	int n = (int)sub.size();

	prop_node_ptr nodes[4];
	copy(sub.begin(), sub.end(), nodes);
	//int target_index = find(nodes, nodes + n, target) - nodes;
	// since we are using 'ordered_subgraph_t', target node always has the largest gid, thus comes first.
	const int target_index = 0;
	int node_without_in = -1;

	pair<int, int> degree_list[4];
	for (int i = 0; i < n; ++i) {
		auto &in = nodes[i]->in;
		for (int j = i + 1; j < n; ++j)
			if (contains(in, nodes[j])) {
				assert(j != target_index);	//*** target should not have out-degree
				degree_list[i].first++;
				degree_list[j].second++;
			}
		if (degree_list[i].first == 0) {
			node_without_in = i;
		}
		//*** shouldn't exist two target nodes
		assert(!(degree_list[i].second == 0 && i != target_index));
	}
	sort(degree_list, degree_list + n);

	// match pattern
	int code = 0;
	for (int i = 0; i < n; ++i) {
		code <<= 4;
		code += degree_list[i].first * 4 + degree_list[i].second;
	}
	auto it = code_to_index.find(code);
	if (it != code_to_index.end()) {
		int i = (code == code_bad && !contains(nodes[0]->in, nodes[node_without_in])) ? PATTERN_BAD : it->second;
		if (b_pos)
			X[i]++;
		else
			Y[i]++;
	}
}

//
// Function: extendSubGraph
//    Extends subgraph 'sub' with one node, compute the new extension node set,
//    and recursively call the function itself to traverse all the pattern instances.
//
// Templete arguments:
//    b_pos - whether it's counting positive pattern or not;
//    SAMPLE_VERTICES - whether should it do vertex sampling, i.e. p=1?
//
// Inputs:
//    sub - nodes (ordered by gid) of a potential pattern;
//    ext - extension set of nodes, neighbors of 'sub'
//
template<bool b_pos, bool SAMPLE_VERTICES>
void extendSubGraph(ordered_subgraph_t &sub, ordered_subgraph_t &ext) {
	// count pattern if there're >= 2 nodes
	if (sub.size() > 1) {
		countPattern<b_pos>(sub);

		if (sub.size() == 4)
			return;
	}

	// Compute new 'ext' only if < 3 nodes,
	// since 'ext' won't be used when there're 4 nodes.
	if (sub.size() < 3) {
		while (!ext.empty()) {
			auto it_w = ext.begin();
			auto w = *it_w;
			ext.erase(it_w);
			if (!SAMPLE_VERTICES || bernoulli_vert[b_pos](gen)) {
				auto ext_prime(ext);
				//for (auto u : w->in) {//if (u->gid < w->gid) {
				//	assert(!contains(sub, u));
				//	ext_prime.insert(u);
				//}
				ext_prime.insert(w->in.begin(), w->in.end());
				it_w = sub.insert(w).first;
				extendSubGraph<b_pos, SAMPLE_VERTICES>(sub, ext_prime);
				sub.erase(it_w);
			}
		}
	}else {	// == 3
		while (!ext.empty()) {
			auto it_w = ext.begin();
			auto w = *it_w;
			ext.erase(it_w);
			if (!SAMPLE_VERTICES || bernoulli_vert[b_pos](gen)) {
				it_w = sub.insert(w).first;
				extendSubGraph<b_pos, SAMPLE_VERTICES>(sub, ext);
				sub.erase(it_w);
			}
		}
	}
}

//
// Function: EnumInfluencePattern
//    This function together with 'extendSubGraph' implements the
//    "Algorithm 2: EnumInfluencePattern", which computes the influence
//    pattern statistics with target node 'target'.
//
// Templete arguments:
//    b_pos - whether it's counting positive pattern or not;
//    SAMPLE_VERTICES - whether should it do vertex sampling, i.e. p=1?
//
// Inputs:
//    target - represents the target node;
//
template<bool b_pos, bool SAMPLE_VERTICES>
void EnumInfluencePattern(prop_node_ptr target) {
	ordered_subgraph_t sub;
	sub.insert(target);
	ordered_subgraph_t ext(target->in.begin(), target->in.end());

	extendSubGraph<b_pos, SAMPLE_VERTICES>(sub, ext);
}

///////////////////////////////////////////////////////////////////
clock_t time0;	// program start time
int N;			// number of users

#ifdef UNDIRECTED_GRAPH
vector< vector<int> > influencer;
#define influencee influencer
#else
// influener[u] is the potential influencers of u, e.g. u follows influener[u]
vector< vector<int> > influencer;
// influenee[u] is the potential influencees of u, e.g. influenee[u] follows u
vector< vector<int> > influencee;
#endif

char *LOG_FILE, *OUT_DIR;
// File of the action log
ifstream f_log;
// 'log_queue' refers to the Log entries that are maintained in memory,
//   i.e. the action diffusion graph, corresponds to 'Q' in the pseudo-code.
//   And only entries within 3*tau ahead current time are kept, (t-3tau,t].
// 'queue_top' points to the first entry within (t-tau,t].
deque< prop_node_ptr > log_queue;
int queue_top;

// Hash map: hash_uid_aid(user_id, action_id) -> list of prop_nodes of this user and action;
// corresponds to 'H' in the pseudo-code.
#ifdef ASSUME_ONE_ACTION_PER_USER
unordered_map< long long, prop_node_ptr > ua_hash(100000);
#else
unordered_map< long long, deque<prop_node_ptr> > ua_hash(200000);
#endif
long long hash_uid_aid(int uid, int aid) {
	return ((long long)aid) * N + uid;
}

// Result correction when edge sampling is enabled.
void correct_edge_result(double *v) {
	double x[PATTERN_COUNT];
	memcpy(x, v, sizeof(x));
	auto p = bernoulli_edge.p();
	v[18] += (p - 1)*x[19];
	v[17] += (p - 1)*x[19];
	v[16] += (p - 1)*x[19];
	v[15] += (p - 1)*x[19];
	v[14] += (p - 1)*x[19];
	v[13] += p*(x[16] + x[17] + p*x[19]) - (v[16] + v[17] + v[19]);
	v[12] += p*(2 * x[14] + 2 * x[17] + 2 * x[18] + p * 3 * x[19]) - (2 * v[14] + 2 * v[17] + 2 * v[18] + 3 * v[19]);
	v[11] += p*(x[15] + x[16] + 2 * x[18] + p * 2 * x[19]) - (v[15] + v[16] + 2 * v[18] + 2 * v[19]);
	v[10] += p*(x[15] + x[16] + p*x[19]) - (v[15] + v[16] + v[19]);
	v[9] += p*(x[14] + x[16] + p*x[19]) - (v[14] + v[16] + v[19]);
	v[8] += p*(x[14] + x[15] + p*x[19]) - (v[14] + v[15] + v[19]);
	v[7] += p*(x[12] + p*(x[14] + x[17] + x[18] + p*x[19])) - (v[12] + v[14] + v[17] + v[18] + v[19]);
	v[6] += p*(x[10] + x[11] + p*(x[15] + x[16] + x[18] + p*x[19])) - (v[10] + v[11] + v[15] + v[16] + v[18] + x[19]);
	v[5] += p*(x[8] + x[9] + x[11] + x[12] + 2 * x[13] + p*(2 * x[14] + x[15] + 2 * x[16] + 2 * x[17] + 2 * x[18] + p * 3 * x[19]))
		- (v[8] + v[9] + v[11] + v[12] + 2 * v[13] + 2 * v[14] + v[15] + 2 * v[16] + 2 * v[17] + 2 * v[18] + 3 * v[19]);
	v[4] += p*(x[8] + x[9] + x[10] + p*(x[14] + x[15] + x[16] + p*x[19])) - (v[8] + v[9] + v[10] + v[14] + v[15] + v[16] + x[19]);
	v[2] += (p - 1)*x[3];
	v[1] += (p - 1)*x[3];
}

void print_result(FILE *runlog_f, bool SAMPLE_EDGES, int COUNT_WHAT) {
	for (int i = 0; i < PATTERN_COUNT; ++i) {
		if (COUNT_WHAT & 1) estX[i] = X[i] * S[1][i];
		if (COUNT_WHAT & 2) estY[i] = Y[i] * S[0][i];
	}
	if (SAMPLE_EDGES) {
		if (COUNT_WHAT & 1) correct_edge_result(estX);
		if (COUNT_WHAT & 2) correct_edge_result(estY);
	}
	if (COUNT_WHAT == 1) {
		for (int i = 0; i < PATTERN_COUNT; ++i) {
			printf("Pt %02d: %lld\t%.0f\n", i+1, X[i], round(estX[i]));
			fprintf(runlog_f, "Pt %02d: %lld\t%.0f\n", i+1, X[i], round(estX[i]));
		}
	}else if (COUNT_WHAT == 2) {
		for (int i = 0; i < PATTERN_COUNT; ++i) {
			printf("Pt %02d: %lld\t%.0f\n", i+1, Y[i], round(estY[i]));
			fprintf(runlog_f, "Pt %02d: %lld\t%.0f\n", i+1, Y[i], round(estY[i]));
		}
	}else {
		for (int i = 0; i < PATTERN_COUNT; ++i) {
			auto inf_prob = estX[i]/(estX[i]+estY[i]);
			printf("Pt %02d: %-10lld\t%-10.0f,\t%-10lld\t%-10.0f,\t%.6f\n", i+1, X[i], round(estX[i]), Y[i], round(estY[i]), inf_prob);
			fprintf(runlog_f, "Pt %02d: %-10lld\t%-10.0f,\t%-10lld\t%-10.0f,\t%.6f\n", i+1, X[i], round(estX[i]), Y[i], round(estY[i]), inf_prob);
		}
	}
}

//
// Function: StructInfCount
//   Implements
//
// Templete arguments:
//    SAMPLE_VERTICES - whether should it do vertex sampling, i.e. p=1?
//    SAMPLE_EDGES - whether should it do edge sampling, i.e. q=1?
//
// Inputs:
//    tau - parameter that specifies the largest time delay on a influence edge;
//    COUNT_WHAT - 1: positive/active patterns, 2: inactive patterns, 3: both active & inactive ones
//
template<bool SAMPLE_VERTICES, bool SAMPLE_EDGES>
void StructInfCount(int tau, int COUNT_WHAT = 3) {
	auto time1 = clock();
	char filename[128];
	sprintf(filename, "%s/runlog_tau=%d_px=%f_py=%f_q=%f_%d.txt",
		OUT_DIR, tau, bernoulli_vert[1].p(), bernoulli_vert[0].p(), bernoulli_edge.p(), rand());
	printf("\ntime: %lld start counting, result = %s\n", time1 - time0, filename);
	FILE *runlog_f = fopen(filename, "w");
	fputs(LOG_FILE, runlog_f);
	fputs(filename, runlog_f);

	queue_top = 0;
	memset(X, 0, sizeof(X));
	if (COUNT_WHAT & 2) memset(Y, 0, sizeof(Y));

	int uid, aid, t = 0;
	int num = 0;

	while (f_log >> uid >> aid >> t) {
		// here comes a new prop_node / log entry (u,a,t)
		
		// Count negative samples when a node is exiting the [t-tau,t] duration queue
		int t0 = t - tau;
		while (queue_top != log_queue.size() && log_queue[queue_top]->time < t0) {
			auto node_popped = log_queue[queue_top];
			++queue_top;

			if (COUNT_WHAT & 2)
				// check if a neighbor is active
				for (auto target_id : influencee[node_popped->uid]) {
					auto h = hash_uid_aid(target_id, node_popped->aid);
					if (!contains(ua_hash, h))
					if (!SAMPLE_VERTICES || bernoulli_vert[false](gen)) {
						// create a negative case (in stack, should run faster)
						prop_node _node_temp;
						auto node_temp = &_node_temp;
						node_temp->time = node_popped->time + tau;
						node_temp->gid = num;

						// add fake propagation edges, which only come from influencers
						for (auto &f : influencer[target_id]) {
							auto it = ua_hash.find(hash_uid_aid(f, node_popped->aid));
							if (it != ua_hash.end()) {
#ifdef ASSUME_ONE_ACTION_PER_USER
								auto p = it->second;
#else
								for (auto p : it->second)
#endif
									// time should be STRICTLY smaller
									if (p->time < node_temp->time && (!SAMPLE_EDGES || bernoulli_edge(gen)))
										node_temp->in.insert(p);
							}
						}

						// traverse pattern instances as usual
						EnumInfluencePattern<false, SAMPLE_VERTICES>(node_temp);

						// remove added edges (note that there're incoming links only)
						//for (auto o : node_temp->in) o->out.erase(node_temp);
					}
				}

			// remove 'node_popped' from hash table
			auto it = ua_hash.find(hash_uid_aid(node_popped->uid, node_popped->aid));
#ifdef ASSUME_ONE_ACTION_PER_USER
			assert(it->second == node_popped);
			ua_hash.erase(it);
#else
			assert(it->second.front() == node_popped);
			it->second.pop_front();
			if (it->second.empty())
				ua_hash.erase(it);
#endif
		}

		// remove out-dated (>3*tau) entries from log queue
		while (!log_queue.empty() && log_queue.front()->time < t0 - tau * 2) {
			delete log_queue.front();
			log_queue.pop_front();
			queue_top--;
		}

		// add the new log entry (as a positive prop_node) to prop_graph
		auto node_new = prop_node_ptr(new prop_node);
		log_queue.push_back(node_new);

		node_new->uid = uid;
		node_new->aid = aid;
		node_new->time = t;
		node_new->gid = num;

		// add propagation edges
		for (auto &f : influencer[uid]) {
			auto it = ua_hash.find(hash_uid_aid(f, aid));
			if (it != ua_hash.end()) {
#ifdef ASSUME_ONE_ACTION_PER_USER
				auto p = it->second;
#else
				for (auto p : it->second)
#endif
					// time should be STRICTLY smaller
					if (p->time < t && (!SAMPLE_EDGES || bernoulli_edge(gen)))
						node_new->in.insert(p);
			}
		}
#ifdef ASSUME_ONE_ACTION_PER_USER
		assert(ua_hash.find(hash_uid_aid(uid, aid)) == ua_hash.end());
		ua_hash[hash_uid_aid(uid, aid)] = node_new;
#else
		ua_hash[hash_uid_aid(uid, aid)].push_back(node_new);
#endif

		// count patterns of this new node
		if ((COUNT_WHAT & 1) && (!SAMPLE_VERTICES || bernoulli_vert[true](gen)))
			EnumInfluencePattern<true, SAMPLE_VERTICES>(node_new);


		if (num % 100000 == 0) {
			printf("time: %lld num = %d queue_size %d hash_size %d\n", clock() - time0,
				num, log_queue.size(), ua_hash.size());
			fprintf(runlog_f, "time: %lld num = %d queue_size %d hash_size %d\n", clock() - time0,
				num, log_queue.size(), ua_hash.size());
			print_result(runlog_f, SAMPLE_EDGES, COUNT_WHAT);
			fflush(runlog_f);
		}
		num++;
	}

	while (!log_queue.empty()) {
		delete log_queue.front();
		log_queue.pop_front();
		//queue_top--;
	}
	ua_hash.clear();

	printf("time of this run: %lld\n", clock() - time1);
	fprintf(runlog_f, "time of this run: %lld\n", clock() - time1);
	print_result(runlog_f, SAMPLE_EDGES, COUNT_WHAT);
	fclose(runlog_f);
}

#ifndef _WIN32
#include <unistd.h>
#endif

int main(int argc, char *argv[]) {
	if (argc <= 4) {
		puts("not enough parameters\nusage: ./structinf GRAPH_FILE LOG_FILE OUT_DIR /tau1,px1,py1,q1/*times1,/tau2,px2,py2,q2/*times2,..."
			"\ne.g.,\n  ./structinf weibo/graph.txt weibo/logs.txt . /90000,0.6,0.1,0.9/*5,/3600,0.7,0,1/"
			"\nPS:\n  '*times' can be omitted, meaning run with the specific parameters defined between /.../ only once;"
			"\n  'px' or 'py' can be 0, meaning not counting active or inactive instances.\n");
		return 0;
	}
#ifndef _WIN32
	srand(time(0) * getpid());
#else
	srand(time(0));
#endif
	char line[100], *GRAPH2, *PARAM;
	int num;
	
	GRAPH2 = argv[1];			// weibo/graph2.txt
	LOG_FILE = argv[2];			// weibo/logs.txt
	OUT_DIR = argv[3];			// .
	PARAM = argv[4];			// /tau,px,py,q/*times...
	printf("GRAPH2=%s, LOG_FILE=%s, OUT_DIR=%s, PARAM=%s\n", GRAPH2, LOG_FILE, OUT_DIR, PARAM);

	calc_pattern_codes();

	time0 = clock();
	// read graph
	{
		FILE *f = fopen(GRAPH2, "r");
		int M = -1, u, k, v;
		fgets(line, sizeof(line), f);
		sscanf(line, "%d %d", &N, &M);
		printf("graph user count = %d, edge count = %d\n", N, M);
		influencer.resize(N);
		influencee.resize(N);
		num = 0;
		while (fscanf(f, "%d %d", &u, &k) != EOF) {
			if (num++ % 100000 == 0) printf("%d / %d\n", num, N);
			influencer[u].reserve(k);
			while (k--) {
				// u follows v, so v has the possibility to influence u
				fscanf(f, "%d", &v);
#ifdef UNDIRECTED_GRAPH
				influencer[u].push_back(v);
#else
				influencer[u].push_back(v);
				influencee[v].push_back(u);
#endif
			}
		}
		fclose(f);
	}
	printf("time: %lld\n", clock() - time0);

	f_log.open(LOG_FILE);

	int tau;
	for (double px, py, q; PARAM && sscanf(PARAM, "/%d,%lf,%lf,%lf/", &tau, &px, &py, &q) != -1;) {
		PARAM = strchr(PARAM + 2, '/') + 1;
		if (PARAM[0] == '*')
			sscanf(PARAM += 1, "%d", &num);
		else
			num = 1;
		PARAM = strchr(PARAM, '/');

		reset_S();
		set_vert_prob(px, py);
		set_edge_prob(q);
	
		for (int i = 0; i < num; ++i) {
			f_log.clear();
			f_log.seekg(0);

			// Invoke the algorithm here.
			// Though ugly, template specializations should be faster than sampling w.prob.1.
			int COUNT_WHAT = 3;
			if (px == 0) COUNT_WHAT -= 1, px = 1;
			if (py == 0) COUNT_WHAT -= 2, py = 1;
			bool sv = px != 1 || py != 1, se = q != 1;
			if (sv && se) StructInfCount<1, 1>(tau, COUNT_WHAT);
			if (sv && !se) StructInfCount<1, 0>(tau, COUNT_WHAT);
			if (!sv && se) StructInfCount<0, 1>(tau, COUNT_WHAT);
			if (!sv && !se) StructInfCount<0, 0>(tau, COUNT_WHAT);
		}
	}
}