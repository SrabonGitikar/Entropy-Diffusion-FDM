#include <bits/stdc++.h>
#include <fstream>

using namespace std;

int main() {
    // 1. Physical and Grid Parameters
    double L = 10.0;          // Total length of our 1D domain
    int NX = 200;             // Number of discrete spatial nodes
    double dx = L / (NX - 1); // Distance between each node
    double D = 1.0;           // Diffusivity constant

    // 2. Time Parameters
    double dt = 0.001;        // Small time step
    double t_max = 2.0;       // How long the simulation runs
    int NT = t_max / dt;      // Total number of time steps to compute

    // 3. Stability Check
    double alpha = D * dt / (pow(dx, 2));
    if (alpha > 0.5) {
        cout << "Warning: alpha > 0.5. Simulation will explode! Decrease dt." << endl;
        return -1;
    }

    // 4. Array Initialization
    // p_current holds the state at time 'n'
    // p_next holds the calculated future state at time 'n+1'
    vector<double> p_current(NX, 0.0);
    vector<double> p_next(NX, 0.0);

    // 5. Initial Condition (Gaussian Distribution)
    double mu = L / 2.0; // Center of the grid
    double sigma = 0.5;  // Width of the blob
    double sum = 0.0;

    for (int i = 0; i < NX; i++) {
        double x = i * dx;
        p_current[i] = exp(-0.5 * pow((x - mu) / sigma, 2));
        sum += p_current[i] * dx; // Riemann sum for area
    }

    // Normalize the array so the total probability equals exactly 1.0
    for (int i = 0; i < NX; i++) {
        p_current[i] /= sum;
    }

    cout << "Initialization complete! Total probability = 1.0" << endl;
    cout << "Alpha (Stability) = " << alpha << endl;

    // TODO: The Main Time-Stepping Loop goes here

    // Open a CSV file to save our data
    ofstream outFile("entropy_data.csv");
    outFile << "Time,Entropy,Fisher_RHS" << endl;

    for (int n = 0; n < NT; n++) {
        // 1. UPDATE THE FLUID (FDM)
        for (int i = 1; i < NX - 1; i++) {
            p_next[i] = p_current[i] + alpha * (p_current[i+1] - 2.0 * p_current[i] + p_current[i-1]);
        }
        
        // Boundary Conditions: Probability is 0 at the extreme walls
        p_next[0] = 0.0;
        p_next[NX - 1] = 0.0;

        // 2. CALCULATE ENTROPY AND FISHER INFORMATION
        double H = 0.0;
        double Fisher_RHS = 0.0;

        for (int i = 1; i < NX - 1; i++) {
            // We use a tiny threshold (1e-12) to avoid log(0) and division by zero at the empty edges
            if (p_next[i] > 1e-12) { 
                // Shannon Entropy integral: H = - p * log(p) * dx
                H += -p_next[i] * log(p_next[i]) * dx;
                
                // Spatial gradient using central difference: dp/dx
                double dp_dx = (p_next[i+1] - p_next[i-1]) / (2.0 * dx);
                
                // Fisher Information integral: D * (dp/dx)^2 / p * dx
                Fisher_RHS += D * pow(dp_dx, 2) / p_next[i] * dx;
            }
        }

        // Save data to the file every 10 steps so we don't bloat the CSV
        if (n % 10 == 0) {
            double current_time = n * dt;
            outFile << current_time << "," << H << "," << Fisher_RHS << endl;
        }

        // 3. SWAP ARRAYS FOR THE NEXT TIME STEP
        p_current = p_next;
    }

    outFile.close();
    cout << "Simulation complete! Data saved to entropy_data.csv" << endl;

    return 0;
}