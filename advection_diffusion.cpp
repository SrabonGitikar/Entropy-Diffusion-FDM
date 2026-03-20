#include <bits/stdc++.h>
using namespace std;

// Simulation parameters
const double L = 5.0;            // Spatial domain [-L, L]
const int N = 1000;              // Number of spatial grid points
const double dx = 2.0 * L / N;   // Spatial step size
const double D = 0.1;            // Diffusion coefficient
const double c = 0.5;            // Compression rate (v(x) = -cx)
const double dt = 0.0001;        // Time step size
const double t_max = 2.5;        // Total simulation time
const int output_freq = 100;     // How often to write to CSV

// Function to calculate velocity at position x
double velocity(double x) {
    return -c * x;
}

int main() {
    vector<double> x(N);
    vector<double> p(N, 0.0);
    vector<double> p_new(N, 0.0);

    // Initialize grid and Gaussian probability distribution
    double sigma = 0.5;
    double sum_p = 0.0;
    for (int i = 0; i < N; ++i) {
        x[i] = -L + i * dx;
        p[i] = exp(-(x[i] * x[i]) / (2.0 * sigma * sigma)) / (sqrt(2.0 * M_PI) * sigma);
        sum_p += p[i] * dx;
    }
    
    // Normalize to ensure total probability is exactly 1
    for (int i = 0; i < N; ++i) {
        p[i] /= sum_p;
    }

    // Open CSV file for output
    ofstream outFile("entropy_data_adv.csv");
    outFile << "Time,dH_dt_Numerical,Kinematic_Sink,Fisher_Information,RHS_Theoretical\n";

    double H_prev = 0.0;
    
    // Initial Entropy Calculation
    for (int i = 1; i < N - 1; ++i) {
        if (p[i] > 1e-12) {
            H_prev -= p[i] * log(p[i]) * dx;
        }
    }

    int num_steps = t_max / dt;
    for (int step = 1; step <= num_steps; ++step) {
        double current_time = step * dt;

        // FDM Update (Explicit FTCS with Central Differencing for Advection)
        for (int i = 1; i < N - 1; ++i) {
            double v_right = velocity(x[i+1]);
            double v_left = velocity(x[i-1]);
            
            // Advective term: - d(pv)/dx using central difference
            double advection = - (p[i+1] * v_right - p[i-1] * v_left) / (2.0 * dx);
            
            // Diffusive term: D * d^2p/dx^2 using central difference
            double diffusion = D * (p[i+1] - 2.0 * p[i] + p[i-1]) / (dx * dx);
            
            p_new[i] = p[i] + dt * (advection + diffusion);
        }

        // Boundary conditions (Strictly zero probability at boundaries)
        p_new[0] = 0.0;
        p_new[N-1] = 0.0;

        // Update probability array
        p = p_new;

        // Calculate metrics periodically
        if (step % output_freq == 0) {
            double H_current = 0.0;
            double fisher_info = 0.0;
            double kinematic_sink = 0.0;

            for (int i = 1; i < N - 1; ++i) {
                if (p[i] > 1e-12) {
                    // Shannon Entropy
                    H_current -= p[i] * log(p[i]) * dx;
                    
                    // Fisher Information integral: D * (dp/dx)^2 / p
                    double dp_dx = (p[i+1] - p[i-1]) / (2.0 * dx);
                    fisher_info += D * (dp_dx * dp_dx) / p[i] * dx;
                    
                    // Kinematic Sink integral: p * div(v)
                    // Since v(x) = -cx, div(v) = -c
                    kinematic_sink += p[i] * (-c) * dx; 
                }
            }

            // Calculate Numerical dH/dt (Backward difference)
            double dH_dt_num = (H_current - H_prev) / (dt * output_freq);
            
            // The theoretical right-hand side of your equation
            double rhs_theoretical = kinematic_sink + fisher_info;

            // Write to CSV
            outFile << fixed << setprecision(6) 
                    << current_time << "," 
                    << dH_dt_num << "," 
                    << kinematic_sink << "," 
                    << fisher_info << "," 
                    << rhs_theoretical << "\n";

            H_prev = H_current;
        }
    }

    outFile.close();
    cout << "Simulation complete. Data saved to entropy_data_adv.csv" << endl;

    return 0;
}