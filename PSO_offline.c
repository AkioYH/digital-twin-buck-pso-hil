#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define MAX_POINTS 100000
#define NUM_PARTICLES 150
#define MAX_ITER 300

typedef struct { double time, iL_exp, vo_exp, pwm; } DataPoint;

/* 4 parâmetros de busca: {RL, RSE, V_d, R_on} */
typedef struct { double RL, RSE, V_d, R_on; } Params;

typedef struct {
    Params norm_pos;
    Params vel;
    Params norm_best_pos;
    double best_error;
    int stuck_count;
} Particle;

double rand_double(double min, double max) { return min + (rand() / (double)RAND_MAX) * (max - min); }
double denormalize(double norm, double min, double max) { return min + norm * (max - min); }

Params get_physical_params(Params norm, Params min, Params max) {
    Params p;

    p.RL   = denormalize(norm.RL,   min.RL,   max.RL);
    p.RSE  = denormalize(norm.RSE,  min.RSE,  max.RSE);
    p.V_d  = denormalize(norm.V_d,  min.V_d,  max.V_d);
    p.R_on = denormalize(norm.R_on, min.R_on, max.R_on);
    return p;
}


double calculate_error(Params p, DataPoint *data, int n_points,
                       double Vin, Params norm,
                       double s_vo, double s_il,
                       double D_effective, double L, double C, double Ro)
{
    /* Inicialização de vC usando equilíbrio DC */
    double vC_sim = data[0].vo_exp * (1.0 + p.RSE / Ro) - data[0].iL_exp * p.RSE;
    double iL_sim = data[0].iL_exp;

    double total_error_rms = 0.0;
    int    evaluated_points = 0;

    double vo_sim_max = -1e30, vo_sim_min =  1e30;
    double vo_exp_max = -1e30, vo_exp_min =  1e30;

    int start_point = (int)(n_points * 0.10);

    for (int k = 0; k < n_points - 1; k++) {
        double dt = data[k+1].time - data[k].time;
        if (dt <= 0 || dt > 0.0001) continue;

        int pwm = (data[k].pwm > 0.5) ? 1 : 0;
        double vo_sim = (vC_sim + p.RSE * iL_sim) / (1.0 + p.RSE / Ro);

        if (k >= start_point) {
            double err_i = (data[k].iL_exp - iL_sim) / s_il;
            double err_v = (data[k].vo_exp - vo_sim)  / s_vo;

            total_error_rms += (12.0 * err_i * err_i) + (3 * err_v * err_v);
            evaluated_points++;

            if (vo_sim > vo_sim_max) vo_sim_max = vo_sim;
            if (vo_sim < vo_sim_min) vo_sim_min = vo_sim;
            if (data[k].vo_exp > vo_exp_max) vo_exp_max = data[k].vo_exp;
            if (data[k].vo_exp < vo_exp_min) vo_exp_min = data[k].vo_exp;
        }

        /* PASSO 1: Queda dinâmica com base em iL_sim (atual) */
        double v_switch1 = (pwm == 1) ? (Vin - iL_sim * p.R_on) : (-p.V_d);

        double diL_dt1 = (v_switch1 - vo_sim - iL_sim * p.RL) / L;
        double dvC_dt1 = (iL_sim - (vo_sim / Ro)) / C;

        /* Predição do próximo estado */
        double iL_sim_pred = iL_sim + (diL_dt1 * dt);
        double vC_sim_pred = vC_sim + (dvC_dt1 * dt);
        double vo_sim_pred = (vC_sim_pred + p.RSE * iL_sim_pred) / (1.0 + p.RSE / Ro);

        /* PASSO 2: Queda dinâmica com base em iL_sim_pred (futuro) */
        double v_switch2 = (pwm == 1) ? (Vin - iL_sim_pred * p.R_on) : (-p.V_d);

        double diL_dt2 = (v_switch2 - vo_sim_pred - iL_sim_pred * p.RL) / L;
        double dvC_dt2 = (iL_sim_pred - (vo_sim_pred / Ro)) / C;

        /* Integração de Euler Modificada (Predictor-Corrector) */
        iL_sim += (diL_dt1 + diL_dt2) * 0.5 * dt;
        vC_sim += (dvC_dt1 + dvC_dt2) * 0.5 * dt;

        if (isnan(total_error_rms)) return 1e20;
    }

    double pp_sim = vo_sim_max - vo_sim_min;
    double pp_exp = vo_exp_max - vo_exp_min;
    double err_pp = 0.0;
    if (pp_exp > 0.001 && evaluated_points > 0) {
        err_pp = (pp_exp - pp_sim) / pp_exp;
        err_pp = err_pp * err_pp;
    }

    /* Balanço DC teórico utilizando o D_effective corrigido pelo tempo morto */
    double Vo_dc_exp = data[n_points - 1].vo_exp;
    double IL_dc_exp = data[n_points - 1].iL_exp;

    double Vo_dc_theory = D_effective * Vin;

    double err_dc = (Vo_dc_exp - Vo_dc_theory) / s_vo;
    double penalty_dc = 3.0 * err_dc * err_dc;


    double error_rms = total_error_rms / (evaluated_points + 1);

    return error_rms + 20.0 * err_pp + penalty_dc;
}

int main() {
    srand(time(NULL));
    double Vin = 100.0;

    double L  = 1e-3;   // 1 mH
    double C  = 235e-6; // 235 uF
    double Ro = 16;     // 16 Ohms

    // Limites: {RL, RSE, V_d, R_on}
    const Params min_p = {0.002, 0.05, 0.05, 0.001};
    const Params max_p = {0.02,  0.5,  0.5, 0.1};

    DataPoint *raw  = (DataPoint*) malloc(sizeof(DataPoint) * MAX_POINTS);
    DataPoint *data = (DataPoint*) malloc(sizeof(DataPoint) * MAX_POINTS);

    FILE *fp = fopen("HIL_260427_162655.csv", "r");
    if (!fp) { printf("Erro ao abrir arquivo!\n"); return 1; }

    char line[1024];
    fgets(line, sizeof(line), fp);
    int raw_count = 0;
    while (fgets(line, sizeof(line), fp) && raw_count < MAX_POINTS) {
        if (sscanf(line, "%lf,%lf,%lf,%lf",
                   &raw[raw_count].time,
                   &raw[raw_count].pwm,
                   &raw[raw_count].vo_exp,
                   &raw[raw_count].iL_exp) == 4) raw_count++;
    }
    fclose(fp);

    int count = 0;
    for (int i = 0; i < raw_count; i++) {
        if (i == 0 || raw[i].time > raw[i-1].time) {
            data[count++] = raw[i];
        }
    }
    free(raw);
    printf("Pontos originais: %d  |  Apos remocao de duplicatas: %d\n", raw_count, count);

    /* Definição dos parâmetros para o cálculo do Tempo Morto */
    double D_real = 0.7;
    double fs = 10000.0;       /* Frequência de chaveamento em Hz */
    double td = 5.0e-6 * 0;        /* Tempo morto em segundos */

    /* Aplicação do conceito: Duty cycle efetivo descontando a transição afetada */
    double D_effective = D_real - (td * fs);

    printf("Duty cycle nominal: %.4f\n", D_real);
    printf("Duty cycle efetivo (calculado com tempo morto): %.4f\n", D_effective);

    double s_vo = 0.001, s_il = 0.001;
    for (int i = 0; i < count; i++) {
        if (fabs(data[i].vo_exp) > s_vo) s_vo = fabs(data[i].vo_exp);
        if (fabs(data[i].iL_exp) > s_il) s_il = fabs(data[i].iL_exp);
    }

    Particle swarm[NUM_PARTICLES];
    Params gbest_norm;
    double gbest_error = 1e30;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        double *n_pos = (double*)&swarm[i].norm_pos;
        for (int j = 0; j < 4; j++) n_pos[j] = rand_double(0.1, 0.9);
        memset(&swarm[i].vel, 0, sizeof(Params));
        swarm[i].stuck_count = 0;
        swarm[i].norm_best_pos = swarm[i].norm_pos;

        Params phys = get_physical_params(swarm[i].norm_pos, min_p, max_p);
        swarm[i].best_error = calculate_error(phys, data, count, Vin, swarm[i].norm_pos, s_vo, s_il, D_effective, L, C, Ro);
        if (swarm[i].best_error < gbest_error) {
            gbest_error = swarm[i].best_error;
            gbest_norm  = swarm[i].norm_pos;
        }
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        double w  = 0.50;
        double c1 = 0.80, c2 = 0.60;
        const double v_max = 0.08;

        for (int i = 0; i < NUM_PARTICLES; i++) {
            double r1 = rand_double(0, 1), r2 = rand_double(0, 1);
            double *p  = (double*)&swarm[i].norm_pos;
            double *v  = (double*)&swarm[i].vel;
            double *bp = (double*)&swarm[i].norm_best_pos;
            double *gp = (double*)&gbest_norm;

            for (int j = 0; j < 4; j++) {
                v[j] = (w * v[j]) + (c1 * r1 * (bp[j] - p[j])) + (c2 * r2 * (gp[j] - p[j]));

                if (v[j] >  v_max) v[j] =  v_max;
                if (v[j] < -v_max) v[j] = -v_max;
                p[j] += v[j];
                if (p[j] < 0.0) { p[j] = 0.001; v[j] *= -0.5; }
                if (p[j] > 1.0) { p[j] = 0.999; v[j] *= -0.5; }
            }

            Params phys = get_physical_params(swarm[i].norm_pos, min_p, max_p);
            double err  = calculate_error(phys, data, count, Vin, swarm[i].norm_pos, s_vo, s_il, D_effective, L, C, Ro);

            if (err < swarm[i].best_error) {
                swarm[i].best_error    = err;
                swarm[i].norm_best_pos = swarm[i].norm_pos;
            }
            if (err < gbest_error) {
                gbest_error = err;
                gbest_norm  = swarm[i].norm_pos;
            }
        }

        if (iter % 20 == 0) {
            Params tmp = get_physical_params(gbest_norm, min_p, max_p);
            printf("Iter %4d | Erro: %.4e | RL=%.4fohm RSE=%.4fohm V_d=%.3fV R_on=%.4fohm\n",
                   iter, gbest_error,
                   tmp.RL, tmp.RSE, tmp.V_d, tmp.R_on);
        }
    }

    Params res = get_physical_params(gbest_norm, min_p, max_p);
    printf("\n--- RESULTADOS FINAIS ---\n");
    printf("RL:     %.5f Ohm\n",           res.RL);
    printf("RSE:    %.5f Ohm\n",           res.RSE);
    printf("V_d:    %.4f V\n",             res.V_d);
    printf("R_on:   %.5f Ohm (%.2f mOhm)\n", res.R_on, res.R_on*1e3);
    printf("Duty cycle efetivo considerado no modelo: %.4f\n", D_effective);

    free(data);
    return 0;
}