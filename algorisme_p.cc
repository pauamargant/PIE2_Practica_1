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

const int r = 2; // number of hidden states

// Probability Matrix for the hidden markov model
const vector<vector<double>> A = {{log(0.95), log(0.05)},
                                  {log(0.075), log(0.925)}};

// Emission probability Matrix
const vector<vector<double>> B = {{log(1 / (double)6), log(1 / (double)6), log(1 / (double)6), log(1 / (double)6), log(1 / (double)6), log(1 / (double)6)},
                                  {log(1 / (double)10), log(1 / (double)10), log(1 / (double)10), log(1 / (double)10), log(1 / (double)10), log(1 / (double)2)}};

// Initial probabilities π
const vector<double> P = {log(0.5), log(0.5)};

//////////////////////////////
// File reading and writing //
//////////////////////////////

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
        S[i++] = s - '0' - 1;
    f.close();
    cout << "File read successfully" << endl;
    return S;
}

/*
Given a vector of integers R, saves its content into output_fname.txt
The contents of the vector are saved as chars without spaces.
*/
void save_states(string output_fname, vector<int> R)
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
hidden states that maximizes the last x_N
*/
vector<int> find_x(const Matrix &M)
{
    int N = M.size();
    vector<int> x(N, -1);
    int max_prob = -inf;

    // we find x_N
    for (int i = 0; i < r; ++i)
        if (M[N - 1][i].prob > max_prob)
        {
            max_prob = M[N - 1][i].prob;
            x[N - 1] = i;
        }

    // we find the rest x_0.....x_{N-1}
    for (int n = N - 2; n >= 0; --n)
        x[n] = M[n + 1][x[n + 1]].prev;

    return x;
}

/*
Given a vector with obsevable states, returns a vector with the hidden states.

A Matrix N*r*2 is used as a dictionary to store the calculations made. In the
position (n, i) there will be stored the State of the observation y_n if the
corresponding dice state were i.
*/
vector<int> HMM(const vector<int> &S)
{
    int N = S.size();
    Matrix M(N, Row(r, State{-inf, -1}));

    // base case
    for (int i = 0; i < r; ++i)
        M[0][i].prob = B[i][S[0]] + P[i];

    // we calculate the maximum probability and psi for all the other states
    for (int n = 1; n < N; ++n)
        for (int i = 0; i < r; ++i)
        {
            int psi = -1;
            M[n][i].prob = B[i][S[n]] + gamma(n, i, M, psi); // hem de fer gamma
            M[n][i].prev = psi;
        }

    return find_x(M);
}

/*
Given an input file name input_fname with the observable states
and the output file name output_fname, creates the ouput_fname.txt
and stores in it the hidden states
*/
void fraudulent_casino(string input_fname, string output_fname)
{
    vector<int> S = read_observable_states(input_fname, 100000);
    vector<int> R = HMM(S);
    cout << "States computed successfully" << endl;
    save_states(output_fname, R);
}

int main()
{
    cout << "Write the observable states file name and the results filename" << endl;
    string input_fname, output_fname;
    cin >> input_fname >> output_fname;
    fraudulent_casino(input_fname, output_fname);
}