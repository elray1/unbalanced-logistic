data {
  int n;
  real x[n];
  int y[n];
}

transformed data {
  real pi_hat[3];

  for(k in 1:3) {
    pi_hat[k] = 0;
    for(i in 1:n) {
      if(y[i] == k) {
        pi_hat[k] += 1;
      }
    }
    pi_hat[k] = pi_hat[k] / n;
  }
}

parameters {
  // beta vector
  real intercept[2];
  real beta[2];
}

transformed parameters {
  matrix[n, 3] log_class_probs;

  {
    real log_denom;
    for(i in 1:n) {
      log_class_probs[i, 1] = 0;
      log_class_probs[i, 2] = intercept[1] + beta[1] * x[i];
      log_class_probs[i, 3] = intercept[2] + beta[2] * x[i];
      log_denom = log_sum_exp(0, log_sum_exp(log_class_probs[i, 2], log_class_probs[i, 3]));
      for(k in 1:3) {
        log_class_probs[i, k] -= log_denom;
      }
    }
  }
}

model {
  for(i in 1:n) {
    target += (1 - pi_hat[y[i]]) * log_class_probs[i, y[i]];
  }
}
