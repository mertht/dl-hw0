#include <assert.h>
#include <math.h>
#include "uwnet.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    // TODO: verify
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            double fx = 0.0;
            if(a == LOGISTIC){
                float exp_val = exp((double) -x);
                fx = 1 / (1 + exp_val);
            } else if (a == RELU){
                fx = (x < 0) ? 0 : x;
            } else if (a == LRELU){
                fx = (x < 0) ? 0.1*x : x;
            } else if (a == SOFTMAX){
                fx = exp((double) x);
            }
            m.data[i*m.cols + j] = fx;
            sum += m.data[i*m.cols + j];
        }
        if (a == SOFTMAX) {
            for (int j = 0; j < m.cols; j++) {
                m.data[i*m.cols + j] /= sum;
            }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    // TODO: verify
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double fx = m.data[i*m.cols + j];
            double dphi = 0.0;
            if(a == LOGISTIC){
                dphi = fx * (1 - fx);
            } else if (a == RELU){
                dphi = (fx < 0) ? 0 : 1;
            } else if (a == LRELU){
                dphi = (fx < 0) ? 0.1 : 1;
            } else if (a == SOFTMAX){
                dphi = 1;
            }
            d.data[i*m.cols + j] *= dphi; // TODO: are we modifying the correct matrix?
        }
    }
}
