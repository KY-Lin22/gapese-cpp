#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gapese_solver_handle gapese_solver_handle;

typedef struct {
    int n_x;
    int n_lambda;
    int n_b;
    int n_d;
} gapese_problem_size;

typedef struct {
    int print_level;
    double dgap_a;
    double dgap_b;
    double fb_sigma;
    int max_iter;
    double tol_kkt;
    double tol_dgap;
    double newton_step_max;
    double mu_min;
    double mu_max;
    double mu_update;
    double nu_x;
    double nu_xi;
    double nu_b;
    double nu_d;
    double beta_init;
    double rho;
    double alpha_min;
    double nu_alpha;
    double nu_D;
} gapese_options;

typedef struct {
    int rows;
    int cols;
    int nnz;
    const int* col_ptr;
    const int* row_idx;
    const double* values;
} gapese_csc_view;

typedef struct {
    int size;
    const double* values;
} gapese_vector_view;

typedef struct {
    gapese_csc_view Q_x, Q_lambda, Q_eta;
    gapese_vector_view r_x, r_lambda, r_eta;
    gapese_csc_view A_x, A_lambda, A_eta;
    gapese_vector_view b;
    gapese_csc_view C_x, C_lambda, C_eta;
    gapese_vector_view d;
} gapese_ccqp_view;

typedef struct {
    gapese_vector_view x, lambda, eta;
    gapese_vector_view v_d, gamma_b, gamma_d;
    int has_v_d;
    int has_gamma_b;
    int has_gamma_d;
} gapese_initial_view;

typedef struct {
    double* x;
    double* lambda;
    double* eta;
    double* v_d;
    double* gamma_b;
    double* gamma_d;
} gapese_solution_buffer;

typedef struct {
    double cost;
    double dgap_res;
    double kkt_res;
    double mu;
    int iter_num;
    int status;
    int line_search_status;
    double time_check_termination;
    double time_search_direction;
    double time_step_size;
    double time_other;
    double time_total;
} gapese_info_c;

gapese_options gapese_default_options(void);
gapese_solver_handle* gapese_create(gapese_problem_size problem, gapese_options options, char* error, int error_capacity);
void gapese_destroy(gapese_solver_handle* handle);
int gapese_solve(
    gapese_solver_handle* handle,
    const gapese_ccqp_view* data,
    const gapese_initial_view* initial,
    gapese_solution_buffer* solution,
    gapese_info_c* info,
    char* message,
    int message_capacity,
    char* error,
    int error_capacity);

#ifdef __cplusplus
}
#endif
