import numpy as np
from lib_robotis_hack import *
import visdom
import time
import itertools
from learners import MovingAverage

vis = visdom.Visdom()
vis.close(win=None)

class Horde:

    def __init__(self, observation_class, approximators, policy_controller, policies,
                 demons, plots, data_folder="data"):
        self.observation_class = observation_class
        self.observations = None
        self.approximators = approximators
        self.plots = plots
        self.data_folder = data_folder
        self.demons = demons
        self.predictions = {}
        self.last_prediction = None
        self.approximations = {}
        self.policy_controller = policy_controller
        self.policies = policies
        self.current_action = None
        self.previous_action = None
        self.ude_average = MovingAverage()

    def log_data(self, data, filename):
        filename = "{}/{}".format(self.data_folder, filename)
        with open(filename, 'ab') as f:
            np.savetxt(f, data, delimiter=",", newline=" ", footer="\n", comments="")

    def get_observations(self):
        self.observations = self.observation_class.get_observations()

    def update_approximations(self):
        for name, approximator in self.approximators.iteritems():
            self.approximations[name] = approximator.approximate(self.observations)
            # print(self.approximations)

    def get_predictions(self):
        for name, demon in self.demons.iteritems():
            self.predictions[name] = demon.predict(self.observations)

    def update_predictions(self):
        for _, demon in self.demons.iteritems():
            demon.update(self.observations, self.policy_controller.current_policy)
            # print("RUPEE: {}".format(demon.rupee_vec))
            # print("UDE: {}".format(demon.ude))

    def update_plots(self):
        for name, plot in self.plots.iteritems():
            if name == "main_plot":
                plot.update([self.predictions["ten_step_on_policy_load_demon"], self.observations["s1_load"]])
                # gamma = 0
                # for name, demon in self.demons.iteritems():
                #     if name == "ten_step_on_policy_load_demon":
                #         gamma = demon.current_gamma
                #         # print("CUMULANT: {}".format(gamma))
                # plot.update([self.predictions["ten_step_on_policy_load_demon"], gamma])
            elif name == "rupee_plot":
                rupee_errors = np.average([demon.rupee_vec for demon in self.demons.values() if getattr(demon, "rupee_vec", False) != False])
                plot.update([rupee_errors])
            elif name == "temperature_plot":
                plot.update([self.observations["s1_temp"]])
            elif name == "load_plot":
                plot.update([self.observations["s1_load"]])
            elif name == "ude_plot":
                ude = np.average([demon.ude for demon in self.demons.values() if getattr(demon, "ude", False) != False])
                plot.update([ude])

    def increase_weights(self):
        for name, demon in self.demons.iteritems():
            off_policy_demons = [
            "ten_step_load_demon",
            "ten_step_temp_demon",
            "predict_temp_demon",
            "five_step_load_demon",
            "five_step_temp_demon",
            "predict_zero_angle_demon",
            "return_to_zero_angle_demon",
            "return_to_two_angle_demon",]
            if name in off_policy_demons:
                demon.w *= 20
                demon.theta *= 20

    def decrease_weights(self):
        for name, demon in self.demons.iteritems():
            off_policy_demons = [
            "ten_step_load_demon",
            "ten_step_temp_demon",
            "predict_temp_demon",
            "five_step_load_demon",
            "five_step_temp_demon",
            "predict_zero_angle_demon",
            "return_to_zero_angle_demon",
            "return_to_two_angle_demon",]
            if name in off_policy_demons:
                demon.w /= 10
                demon.theta /= 10

    # def check_pavlov(self):
    #     pass

    # def take_action(self):
    #     pass

    def run(self, runtime):
        if runtime == None:
            runtime = itertools.count()
        else:
            runtime = range(runtime)
        
        for t in runtime:
            start_time = time.time()
            self.get_observations()
            self.update_approximations()
            self.get_predictions()
            self.current_action = self.policy_controller.get_policy(self.observations, self.predictions).get_action()
            # print(self.predictions)
            # self.check_pavlov()
            # self.take_action()
            self.update_predictions()
            # self.log_data(self.predictions, "predictions")
            # self.log_data(self.observations, "observations")
            self.update_plots()
            time_length = time.time() - start_time
            print(abs(time_length))
            if t == 7000:
                self.increase_weights()


# if __name__ == '__main__':
#     h = Horde()
#     h.run()