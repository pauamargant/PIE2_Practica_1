#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <limits>

using namespace std;

/////////////////////////////
// PRÀCTICA 1 PIE - PART 2 //
//                         //
//   Benet Ramió           //
//   Pau Amargant          //
//                         //
/////////////////////////////

////////////////////
// Global variables
////////////////////

double inf = numeric_limits<double>::max();
// We use an struct in order to store together gamma and psi
// for each state i at position n,
struct State
{
    double prob; // Gamma_{n, i} γ
    int prev;    // Psi_{n, i} Ψ
};

using Row = vector<State>;
using Matrix = vector<Row>;

// We declare some necessary constants and the probabilty matrices which
// describe the HMM problem λ = ( A, B, π ). We use log-probabilities in
// order not to have underflow problems.

const int r = 8; // number of hidden states

// Probability Matrix for the hidden markov model
const vector<vector<double>> A = {
    {log(0.1772240), log(0.2682517), log(0.4170629), log(0.1174825), log(0.0035964), log(0.0054745), log(0.0085104), log(0.0023976)},
    {log(0.1672439), log(0.3609201), log(0.2679840), log(0.1838722), log(0.0034131), log(0.0073453), log(0.0054690), log(0.0037524)},
    {log(0.1576226), log(0.3318881), log(0.3681328), log(0.1223776), log(0.0032167), log(0.0067732), log(0.0074915), log(0.0024975)},
    {log(0.0773429), log(0.3475514), log(0.3759440), log(0.1791818), log(0.0015784), log(0.0070929), log(0.0076723), log(0.0036363)},
    {log(0.0002997), log(0.0002047), log(0.0002837), log(0.0002097), log(0.3004009), log(0.2045904), log(0.2844305), log(0.2095804)},
    {log(0.0003216), log(0.0002977), log(0.0000769), log(0.0003016), log(0.3213566), log(0.2984045), log(0.0778445), log(0.3013966)},
    {log(0.0001768), log(0.0002387), log(0.0002917), log(0.0002917), log(0.1766463), log(0.2385228), log(0.2924165), log(0.2914155)},
    {log(0.0002477), log(0.0002457), log(0.0002977), log(0.0002077), log(0.2475044), log(0.2455084), log(0.2984035), log(0.2075849)}};

// Emission probability Matrix
const double l_one = log((double)1);
const double l_zero = log((double)0);

const vector<vector<double>> B = {
    {l_one, l_zero, l_zero, l_zero},
    {l_zero, l_one, l_zero, l_zero},
    {l_zero, l_zero, l_one, l_zero},
    {l_zero, l_zero, l_zero, l_one},
    {l_one, l_zero, l_zero, l_zero},
    {l_zero, l_one, l_zero, l_zero},
    {l_zero, l_zero, l_one, l_zero},
    {l_zero, l_zero, l_zero, l_one}};

// Initial probabilities π
const vector<double> P = {log(1 / (double)8), log(1 / (double)8), log(1 / (double)8), log(1 / (double)8), log(1 / (double)8), log(1 / (double)8), log(1 / (double)8), log(1 / (double)8)};

//////////////////////////////
// File reading and writing //
//////////////////////////////

/*
Transforms nucleobases into its corresponding integer
*/
int transform(char s)
{
    if (s == 'A')
        return 0;
    else if (s == 'C')
        return 1;
    else if (s == 'G')
        return 2;
    else if (s == 'T')
        return 3;
    return -1;
}

/*
Given an input file name input_fname and its number of characters n,
reads it and returns the observations Y^n into a vector.
The observable states in the input file consists of chars without separation
between them.
*/
vector<int> read_observable_states(string input_fname, int n)
{
    ifstream f(input_fname);
    char s;
    vector<int> S(n);
    int i = 0;
    cout << "Reading data from " << input_fname << endl;
    while (f >> s)
    {
        S[i++] = transform(s);
    }
    f.close();
    cout << "File read successfully" << endl;
    return S;
}

/*
Given a vector of integers R, saves its content into output_fname.txt
The contents of the vector are saved as chars without spaces.
*/

void save_states(string output_fname, const vector<int> &R)
{
    cout << "Saving data into " << output_fname << endl;
    ofstream f(output_fname);
    for (int a : R)
        f << a;
    f.close();
    cout << "File saved successfully" << endl;
}

/*
Given the current throw n, a state i, and the matrix M with the
n-1 States already calculated, calculates gamma_n,i (which is the
maximum probability of the state i given a previous state j) and
stores j into psi (the previous state to maximize the probability)
*/
double gamma(int n, int i, const Matrix &M, int &psi)
{

    double max_prob = -inf;
    for (int j = 0; j < r; ++j)
    {
        double p = A[j][i] + M[n - 1][j].prob;
        if (p > max_prob)
        {
            max_prob = p;
            psi = j;
        }
    }
    return max_prob;
}

/*
Given a Matrix M filled with States returns a vector with the path of
hidden states that maximizes the last x_N. It converts the path into ones
and zeros depending on whether it is a CPG-Island.
*/
vector<int> find_x(const Matrix &M)
{
    int N = M.size();
    vector<int> x(N, -1);
    int max_prob = -inf;

    int prev_state;
    // we find x_N
    for (int i = 0; i < r; ++i)
        if (M[N - 1][i].prob > max_prob)
        {
            max_prob = M[N - 1][i].prob;
            x[N - 1] = (i < 4 ? 1 : 0);
            prev_state = i;
        }

    // we find the rest x_0.....x_{N-1}
    for (int n = N - 2; n >= 0; --n)
    {
        prev_state = M[n + 1][prev_state].prev;
        x[n] = (prev_state < 4 ? 1 : 0);
    }
    return x;
}

/*
Given a vector with obsevable states, returns a vector with the hidden states.

A Matrix N*r*2 is used as a dictionary to store the calculations made. In the
position (n, i) there will be stored the State of the observation y_n if the
corresponding dice state were i.
*/
vector<int> HMM(vector<int> S)
{
    int N = S.size();
    Matrix M(N, Row(r, State{-inf, -1}));
    // base case
    for (int i = 0; i < r; ++i)
    {
        M[0][i].prob = B[i][S[0]] + P[i];
    }
    // we calculate the maximum probability and psi for all the other states
    for (int n = 1; n < N; ++n)
    {
        for (int i = 0; i < r; ++i)
        {
            int psi = -1;
            M[n][i].prob = B[i][S[n]] + gamma(n, i, M, psi);
            M[n][i].prev = psi;
        }
    }
    return find_x(M);
}

/*
Given an input file name input_fname with the observable states
and the output file name output_fname, creates the ouput_fname.txt
and stores in it the hidden states
*/
void hmm_analysis(string input_fname, string output_fname)
{
    vector<int> S = read_observable_states(input_fname, 2590);
    vector<int> R = HMM(S);
    save_states(output_fname, R);
}

int main()
{
    cout << "Write the observable states file name and the results filename" << endl;
    string input_fname, output_fname;
    cin >> input_fname >> output_fname;
    hmm_analysis(input_fname, output_fname);
}