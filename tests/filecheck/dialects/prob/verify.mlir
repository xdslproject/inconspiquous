// RUN: quopt %s --verify-diagnostics --split-input-file | filecheck %s

%0 = prob.bernoulli -1.0
// CHECK: Property 'prob' = -1.0 should be in the range [0, 1]

// -----

%1 = prob.bernoulli 1.5
// CHECK: Property 'prob' = 1.5 should be in the range [0, 1]
