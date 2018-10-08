#include <assert.h>
#include <math.h>
#include "uwnet.h"

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            double result = 0.0;
            if(a == LOGISTIC){
                float exp_val = exp((double) -x);
                result = 1 / (1 + exp_val);
            } else if (a == RELU){
                if (x < 0) {
                    result = 0;
                } else {
                    result = x;
                }
            } else if (a == LRELU){
                if (x < 0) {
                    result = -0.1 * x; // TODO: what value to use for leaky
                } else {
                    result = x;
                }
            } else if (a == SOFTMAX){
                result = exp((double) x);
            }
            m.data[i*m.cols + j] = result;
            sum += m.data[i*m.cols + j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
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
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i*m.cols + j];
            // TODO: multiply the correct element of d by the gradient
        }
    }
}
