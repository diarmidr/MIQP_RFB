# coding: utf-8
"""This script performs an NLP optimisation of RFB schedule for a 24h period, and repeats this process for consecutive
24 row blocks in the CSV file. """
#########################
# Import required tools #
#########################

# import off-the-shelf scripting modules
from __future__ import division     # Without this, rounding errors occur in python 2.7, but apparently not in 3.4
import pandas as pd                 # CSV handling
import datetime                     # For time stamping optimisation process

# import PYOMO

from pyomo.environ import *
from pyomo.opt import SolverFactory

years = [2017]
print(datetime.datetime.now())      # Time stamp start of optimisation
for i in years:
    raw_data = pd.read_csv('clean_n2ex_' + str(i) + '_hourly.csv')
    price_data = raw_data.ix[:, 'price_sterling'].tolist()
    price = [float(x) for x in price_data]
    daily_revenue_list = []   # Empty receptacle for daily revenue
    # Below loop goes through data 24h at a time
    for i in range(int(len(price)/24)):
        price_in_period = price[i*24:(i+1)*24]
        ############################
        # PYOMO model starts here. #
        ############################

        model = ConcreteModel()

        #####################
        # Scalar parameters #
        #####################
        # Intrinsic parameters (from Reed et al. 2016, except VOCV50SOC and LossBOP, from Kim 2011 and Weber 2013.
        model.VOCV50SOC = Param(initialize=1.46)  # OCV at 50% SOC (PNNL)
        model.LossBOP = Param(initialize=0.02)  # One way fractional power losses due to BOP, Weber et al. 2013
        model.EffCoul = Param(initialize=0.975)  # Assumed constant with I
        model.ASR = Param(initialize=0.000052)  # Ohm.m2 from linear fit on over-potential v current density
        model.Vfaradaic = Param(initialize=0.02)  # Faradaic over-potential, y axis intercept of linear fit.
        model.BESSMinCurrentDensity = Param(initialize=0)  # This is the minimum current when RFB is active
        model.BESSMaxCurrentDensity = Param(initialize=3200)  # Maximum at which polarization curve is still linear
        model.LossCoul = Param(initialize=36)  # Areal coulombic loss at middle of studied range (A/m2)
        model.PumpPower = Param(initialize=15)  # Pump power in W.

        # RFB system sizing for absolute output of scheduling problem is done as follows:

        # 1: Define power rating (kW)
        BESSRatedPower = 1

        # 2: From experimental data, find VE that gives EffDC ( = EffV*EffCoul*(1-LossBOP)^2 ) >= 0.75 (round-trip)
        EffVRated = 0.801

        # 3: Identify current density that corresponds to above EffV
        BESSRatedCurrentDensity = 2188  # A/m2, current reaching external stack terminals

        # 4: Set stack area such that system can output 1kW DC despite voltaic and balance of plant losses.
        model.StackArea = Param(initialize=1000 * BESSRatedPower / (BESSRatedCurrentDensity * model.VOCV50SOC *
                                                                    sqrt(EffVRated) * (1 - model.LossBOP)))

        # 5: Set energy to power ratio (a.k.a discharge time)
        EtoP = 4

        # 6: Calculate required coulombic capacity, considering coulombic losses during discharge.
        model.BESSCapacity = Param(initialize=BESSRatedCurrentDensity * EtoP * model.StackArea / sqrt(model.EffCoul))


        # Data-set
        model.TimeStep = Param(initialize=1)  # Hours

        ######################
        # indexed parameters #
        ######################

        # Index
        model.I = range(len(price_in_period))

        # Input price data

        def init_price(model, t):
            return price_in_period[t]


        model.Price = Param(model.I, initialize=init_price)


        #  For constraints

        # These generate new parameters to be used in constraint on RFB current-density.
        # The min current density is subtracted from the max, as the min will be re-added
        # If the indicator variable model.Pumping is 1.


        def max_discharge_current_density_bess_init(model, i):
            return model.BESSMaxCurrentDensity
        model.MaxDischargeCurrentDensityBESSIndexed = Param(model.I, initialize=max_discharge_current_density_bess_init)


        def max_charge_current_density_bess_init(model, i):
            return model.BESSMaxCurrentDensity
        model.MaxChargeCurrentDensityBESSIndexed = Param(model.I, initialize=max_charge_current_density_bess_init)


        def min_soc_bess_init(model, i):
            return 0.15
        model.MinSOCBESSIndexed = Param(model.I, initialize=min_soc_bess_init)


        def max_soc_bess_init(model, i):
            return 0.85
        model.MaxSOCBESSIndexed = Param(model.I, initialize=max_soc_bess_init)


        #############
        # variables #
        #############

        def current_density_bess_charge_init(model, i):
            return 800  # In A/m2
        model.CurrentDensityCharge = Var(model.I, within=NonNegativeReals, initialize=current_density_bess_charge_init)


        def current_density_bess_discharge_init(model, i):
            return 800  # In A/m2
        model.CurrentDensityDischarge = Var(model.I, within=NonNegativeReals,
                                            initialize=current_density_bess_charge_init)

        model.RFBActive = Var(model.I, within=Binary, initialize=1)

        ###############
        # constraints #
        ###############

        # The below loop generates a list of SOC values to which constraints are subsequently applied
        model.SOCAtStart = Param(initialize=0.15)
        SOCtracker = []
        SOC = model.SOCAtStart
        for i in model.I:
            SOC_increment = model.StackArea * model.TimeStep \
                            * ((model.CurrentDensityCharge[i])
                               - (model.CurrentDensityDischarge[i]) -
                               (model.RFBActive[i]) * model.LossCoul) / model.BESSCapacity
            SOC = SOC + SOC_increment
            SOCtracker = SOCtracker + [SOC]  # This appends the ith SOC value to the list


        # These constraints are defined w.r.t to the above set of SOC values

        def min_soc_rule(model, t):
            return SOCtracker[t] - model.MinSOCBESSIndexed[t] >= 0
        model.MinSOCBESSConstraint = Constraint(model.I, rule=min_soc_rule)

        def max_soc_rule(model, t):
            return SOCtracker[t] - model.MaxSOCBESSIndexed[t] <= 0
        model.MaxSOCBESSConstraint = Constraint(model.I, rule=max_soc_rule)

        model.SOCAtEnd = SOCtracker[len(model.I) - 1]
        model.BESSEnergyConservation = Constraint(expr=model.SOCAtStart - model.SOCAtEnd == 0)


        # The below constraints encode the condition that no current can be drawn if the RFB is idle.

        def charge_on_off_rule(model, t):
            return model.CurrentDensityCharge[t] - model.MaxChargeCurrentDensityBESSIndexed[t] * model.RFBActive[t] <= 0
        model.ChargeOnOffConstraint = Constraint(model.I, rule=charge_on_off_rule)


        def discharge_on_off_rule(model, t):
            return model.CurrentDensityDischarge[t] - model.MaxDischargeCurrentDensityBESSIndexed[t] * model.RFBActive[
                t] <= 0
        model.DischargeOnOffConstraint = Constraint(model.I, rule=discharge_on_off_rule)


        #############
        # objective #
        #############

        # Units note: division by 10^6 is to go from Wh to MWh, which is price unit.

        def objective_expression(model):
            return model.TimeStep * (1 / 1000000) \
                   * sum(model.Price[t] *
                         model.StackArea *
                         ((model.VOCV50SOC - model.Vfaradaic) * model.CurrentDensityDischarge[t] -
                          (model.VOCV50SOC + model.Vfaradaic) * model.CurrentDensityCharge[t] -
                          (model.CurrentDensityDischarge[t] ** 2 + model.CurrentDensityCharge[t] ** 2) * model.ASR
                          ) - model.Price[t] * (model.RFBActive[t] * model.PumpPower) for t in model.I)


        model.Objective = Objective(rule=objective_expression, sense=maximize)

        opt = SolverFactory('gurobi')

        results = opt.solve(model)

        ##################################################
        # Terminal window output of optimization results #
        ##################################################
        daily_revenue = value(model.Objective)
        print('Daily revenue: £', daily_revenue)
        daily_revenue_list = daily_revenue_list + [daily_revenue]

    #Writes list of daily revenue for year i to a CSV file
    daily_revenue_output = pd.DataFrame({"Revenue, £": daily_revenue_list})
    daily_revenue_output.to_csv(str(int(i)) + "daily_revenue_NLP_IDD2s.csv", sep=',')
    annual_revenue = sum(daily_revenue_list)
    print("Annual revenue: £", str(annual_revenue))
print(datetime.datetime.now())

