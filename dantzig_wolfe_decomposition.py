#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from itertools import count

# minimize   : x1 - 3x2
# subject to : -x1 + 2x2 <= 6    Complicating constraint(s)
#            : x1  + x2  <= 5    Easy constraint(s)
#            x1, x2 >= 0

# initial columns : [0, 0], [0, 5], [5, 0]
# we will start off by considering only first 2 columns

TOL = -1e-6

A = [-1, 2]  # constraint coefficients of first constraint
c = [1, -3]  # objective functions coefficients (take transpose later)
D = [1, 1]  # constraint coefficients of second constraint
d = 5  # RHS of second constraint


xp1 = [0, 0]
xp2 = [0, 5]  # take transpose of both
# first 2 columns

columns = [xp1, xp2]

Asub = [
    [np.dot(A, np.transpose(xp1)), np.dot(A, np.transpose(xp2))],
    [1, 1],
]  # take transpose
csub = [np.dot(np.transpose(c), xp1), np.dot(np.transpose(c), xp2)]

# build a restricted master problem with first 2 constraints only:
model = gp.Model("RMP")
lambdas = model.addVars(len(csub), obj=csub, vtype=GRB.CONTINUOUS)
# first constraint
c1 = model.addConstr(
    gp.quicksum(Asub[0][i] * lambdas[i] for i in range(len(csub))) <= 6, name="c1"
)
# sum of lamdas <= 1
c2 = model.addConstr(
    gp.quicksum(Asub[1][i] * lambdas[i] for i in range(len(csub))) <= 1, name="c2"
)

model.ModelSense = GRB.MINIMIZE


for iter_count in count():
    model.optimize()

    pi = c1.pi
    alpha = c2.pi

    # pricing sub-problem (column generation sub problem : CGSP)
    obj_coeff = c - (np.transpose(A) * pi)
    sub_model = gp.Model("CGSP")
    sub_model.ModelSense = GRB.MINIMIZE
    # find the column that has the most negative reduced cost
    # x_s is the new column
    x_s = sub_model.addVars(2, obj=obj_coeff, vtype=GRB.INTEGER)
    # second constraint (easy constraint)
    sub_model.addConstr(gp.quicksum(x_s) <= 5)

    sub_model.optimize()
    barcs = sub_model.objVal
    barcs = barcs - alpha  # reduced cost of the column

    if barcs < TOL:
        columns.append([x_s[i].X for i in x_s])

        new_obj_coeff = np.dot(np.transpose(c), [x_s[i].X for i in x_s])
        new_constr_coeff = np.dot(A, np.transpose([x_s[i].X for i in x_s]))

        lambdas[len(csub) + iter_count] = model.addVar(
            obj=new_obj_coeff,
            vtype=GRB.CONTINUOUS,
            column=gp.Column([new_constr_coeff, 1], [c1, c2]),
        )
    else:
        print("Objective function value = " + str(model.objVal))
        break

# extract final values of lambdas:
lambdas = [lambdas[i].X for i in lambdas]

# weighted average of lambdas and the columns (x_p)
final_x = [
    sum(columns[i][0] * lambdas[i] for i in range(len(lambdas))),
    sum(columns[i][1] * lambdas[i] for i in range(len(lambdas))),
]
