from approximators import BinnedApproximator
from lib_robotis_hack import *
from horde import Horde
import visdom
import numpy as np
from learners import GTDLambda, TDLambda
from visualizers import VisdomWindowedLine
import time

D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU",baudrate=1000000)
s_list = find_servos(D)
s1 = Robotis_Servo(D,s_list[0])
s2 = Robotis_Servo(D,s_list[1])


# Define Observations
class DynamixelObservations:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.state = {}
        self.get_observations()

    def get_observations(self):
        self.state = {
            "s1_load" : self.s1.read_load(),
            "s1_temp" : self.s1.read_temperature(),
            "s1_angle" : self.s1.read_angle(),
            # "s1_voltage" : self.s1.read_voltage(),
            # "s1_encoder" : self.s1.read_encoder(),
            # "s1_is_moving" : s1.is_moving(),
            # "s2_load" : self.s2.read_load(),
            # "s2_temp" : self.s2.read_temperature(),
            # "s2_angle" : self.s2.read_angle(),
            # "s2_voltage" : self.s2.read_voltage(),
            # "s2_encoder" : self.s2.read_encoder(),
            # "s2_is_moving" : self.s2.is_moving(),
        }
        return self.state

# Define Approximators
binned_approximator = BinnedApproximator([-2.1, 2.1], 15)
class Approximator:
    def __init__(self, approximator):
        self.angle_approximator = approximator
        self.current_approximation = np.zeros(self.angle_approximator.shape * 2)

    def approximate(self, state):
        # bins = np.array(np.digitize(state['s1_angle'], self.angle_approximator.activations))
        bins = self.angle_approximator.approximate(state["s1_angle"])
        if np.sign(state["s1_load"]) > 0:
            bins = np.add(bins, self.angle_approximator.shape)
        # if np.sign(state["s1_load"]) < 0:
        #     bins = np.append(np.copy(bins), np.array(-2))
        self.current_approximation = np.copy(bins)
        return self.current_approximation

    def approximate_no_save(self, state):
        # bins = np.array(np.digitize(state['s1_angle'], self.angle_approximator.activations))
        bins = self.angle_approximator.approximate(state["s1_angle"])
        if np.sign(state["s1_load"]) > 0:
            bins = np.add(bins, self.angle_approximator.shape)
        # if np.sign(state["s1_load"]) < 0:
        #     bins = np.append(np.copy(bins), np.array(-2))
        return np.copy(bins)

    @property
    def shape(self):
        return self.angle_approximator.shape * 2

approximator = Approximator(binned_approximator)

# Cumulant Functions
def zero_angle_cumulant(state):
    return 1

def two_angle_cumulant(state):
    return 1

def load_cumulant(state):
    return state['s1_load']

def temp_cumulant(state):
    return state['s1_temp']

def absolute_load_cumulant(state):
    return abs(state['s1_load'])

# Gamma Functions
def zero_gamma(state):
    return 0

def zero_angle_gamma(state):
    if state == None:
        return 0
    angle = state['s1_angle']
    if np.isclose([angle], [0.0], atol=0.2)[0]:
        return 0.0
    else:
        return 1.0
    # return gamma_zero_angle

def ten_step_gamma(state):
    return 0.9

def five_step_gamma(state):
    return 0.8

def two_angle_gamma(state):
    if state == None:
        return 0
    angle = state['s1_angle']
    if np.isclose([angle], [2.0], atol=0.2)[0]:
        return 0.0
    else:
        return 1.0


# Policy Functions

class SideToSidePolicy:

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.current_action = 1

    def get_action(self):
        if not self.s1.is_moving():
            self.s1.move_angle(2, blocking=False)
            self.current_action = 1

        if np.isclose([self.s1.read_angle()], [-2], atol=0.1)[0]:
            self.s1.move_angle(2, blocking=False)
            self.current_action = 1

        if np.isclose([self.s1.read_angle()], [2], atol=0.1)[0]:
            self.s1.move_angle(-2, blocking=False)
            self.current_action = 2

        return self.current_action

    def action_probability(self, state):
        return 1

class DoNothingPolicy:

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def get_action(self):
        self.s1.move_angle(self.s1.read_angle())
        self.s1.disable_torque()
        time.sleep(1)
        return 0

    def action_probability(self, state):
        return 0

class ReturnToZeroPolicy:

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def get_action(self):
        self.s1.move_angle(0, blocking=False)
        self.current_action = 2
        return self.current_action

    def action_probability(self, state):
        if state['s1_angle'] > 0 and np.sign(state["s1_load"]) < 0:
            return 0
        if state['s1_angle'] < 0 and np.sign(state["s1_load"]) > 0:
            return 0
        return 0.5

class ReturnToAngleTwoPolicy:

    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s1

    def get_action(self):
        self.s1.move_angle(2, blocking=False)
        self.current_action = 2
        return self.current_action

    def action_probability(self, state):
        if state['s1_angle'] < 2 and np.sign(state["s1_load"]) < 0:
            return 0
        # if state['s1_angle'] < 0 and np.sign(state["s1_load"]) > 0:
        #     return 0.5
        return 0.5

# Policy Controller
class MainPolicyController:

    def __init__(self, policies, pavlov_functions):
        self.policies = policies
        self.current_policy = None
        self.pavlov_functions = pavlov_functions
    
    def get_policy(self, state, predictions):
        for _, func in self.pavlov_functions.iteritems():
            if func(predictions):
                self.current_policy = self.policies["do_nothing_policy"]
                return self.current_policy
        self.current_policy = self.policies["side_to_side_policy"]
        return self.current_policy

# Pavlov Functions
def prevent_overload(predictions):
    if predictions["ten_step_on_policy_load_demon"] > 850:
        return True
    return False

def prevent_overload_hard_coded(observations):
    if state["s1_load"] > 200:
        return True
    return False

# Plots
main_plot = VisdomWindowedLine(num_lines=2,
                               line_data=dict(
                                        showlegend=True,
                                        width=1200,
                                        height=220,
                                        marginleft=10,
                                        marginright=10,
                                        marginbottom=10,
                                        margintop=10,
                                        xlabel='Time Step',
                                        ylabel='Reading',
                                        title='Absolute predicted load over 10 steps',
                                        legend=["Predicted Load (N.m)", "Actual Load (N.m)"]
                                    ),
                                )
rupee_plot = VisdomWindowedLine(num_lines=1, window_size=1000,
                                line_data=dict(
                                        showlegend=True,
                                        width=1200,
                                        height=220,
                                        marginleft=10,
                                        marginright=10,
                                        marginbottom=10,
                                        margintop=10,
                                        xlabel='Time Step',
                                        ylabel='RUPEE VALUE',
                                        title='RUPEE MEASURE',
                                        legend=['RUPEE']
                                    ),
                                )
# temperature_plot = VisdomWindowedLine(num_lines=1)
# load_plot = VisdomWindowedLine(num_lines=1)
ude_plot = VisdomWindowedLine(num_lines=1, window_size=1000,
                              line_data=dict(
                                        showlegend=True,
                                        width=1200,
                                        height=220,
                                        marginleft=10,
                                        marginright=10,
                                        marginbottom=10,
                                        margintop=10,
                                        xlabel='Time Step',
                                        ylabel='UDE VALUE',
                                        title='UDE MEASURE',
                                        legend=['UDE']
                                    ),
                                )

plots = {
         "main_plot" : main_plot,
         "rupee_plot" : rupee_plot,
         # "temperature_plot" : temperature_plot,
         "ude_plot" : ude_plot,
         }

observations = DynamixelObservations(s1, s2)

approximators = {"bins_angle_approximator" : approximator}

side_to_side_policy = SideToSidePolicy(s1, s2)
do_nothing_policy = DoNothingPolicy(s1, s2)
return_to_zero_policy = ReturnToZeroPolicy(s1, s2)
return_to_two_angle_policy = ReturnToAngleTwoPolicy(s1, s2)

policies = {"side_to_side_policy" : side_to_side_policy,
            "do_nothing_policy" : do_nothing_policy,
            "return_to_zero_policy" : return_to_zero_policy,
            "return_to_two_angle_policy" : return_to_two_angle_policy}

pavlov_functions = {"prevent_overload" : prevent_overload}

policy_controller = MainPolicyController(policies, pavlov_functions)

# Demons
predict_load_demon = GTDLambda(alpha=0.05, beta=0.001, lambd=0.9, gamma=zero_gamma, cumulant_function=load_cumulant,
                               function_approximator=approximator, policy=return_to_zero_policy,
                               track_rupee=True, track_ude=True, name="predict_load")

predict_zero_on_policy = TDLambda(alpha=0.05, lambd=0.97, gamma=zero_angle_gamma, cumulant_function=zero_angle_cumulant,
                                  function_approximator=approximator, policy=side_to_side_policy, name="predict_zero_on_policy")

ten_step_on_policy_load_demon = TDLambda(alpha=0.1, lambd=0.97, gamma=ten_step_gamma, cumulant_function=absolute_load_cumulant,
                                    function_approximator=approximator, policy=side_to_side_policy, name="ten_step_on_policy_demon")

ten_step_load_demon = GTDLambda(alpha=0.005, beta=0.0001, lambd=0.97, gamma=ten_step_gamma, cumulant_function=absolute_load_cumulant,
                                function_approximator=approximator, policy=return_to_zero_policy,
                                track_rupee=True, track_ude=True, name="predict_load")

ten_step_temp_demon = GTDLambda(alpha=0.001, beta=0.0001, lambd=0.9, gamma=ten_step_gamma, cumulant_function=temp_cumulant,
                               function_approximator=approximator, policy=return_to_zero_policy,
                               track_rupee=True, track_ude=True, name="predict_load")

predict_temp_demon = GTDLambda(alpha=0.001, beta=0.0001, lambd=0.9, gamma=zero_gamma, cumulant_function=temp_cumulant,
                               function_approximator=approximator, policy=return_to_zero_policy,
                               track_rupee=True, track_ude=True, name="predict_load")

five_step_load_demon = GTDLambda(alpha=0.005, beta=0.0001, lambd=0.97, gamma=five_step_gamma, cumulant_function=absolute_load_cumulant,
                                function_approximator=approximator, policy=return_to_zero_policy,
                                track_rupee=True, track_ude=True, name="predict_load")

five_step_temp_demon = GTDLambda(alpha=0.005, beta=0.0001, lambd=0.97, gamma=five_step_gamma, cumulant_function=temp_cumulant,
                                function_approximator=approximator, policy=return_to_zero_policy,
                                track_rupee=True, track_ude=True, name="predict_load")

predict_zero_angle_demon = GTDLambda(alpha=0.005, beta=0.0001, lambd=0.97, gamma=zero_angle_gamma, cumulant_function=zero_angle_cumulant,
                               function_approximator=approximator, policy=side_to_side_policy,
                               track_rupee=True, track_ude=True, name="predict_zero_angle")

return_to_zero_angle_demon = GTDLambda(alpha=0.005, beta=0.0001, lambd=0.97, gamma=zero_angle_gamma, cumulant_function=zero_angle_cumulant,
                               function_approximator=approximator, policy=return_to_zero_policy,
                               track_rupee=True, track_ude=True, name="predict_zero_angle")

return_to_two_angle_demon = GTDLambda(alpha=0.005, beta=0.0001, lambd=0.97, gamma=two_angle_gamma, cumulant_function=two_angle_cumulant,
                               function_approximator=approximator, policy=return_to_two_angle_policy,
                               track_rupee=True, track_ude=True, name="predict_zero_angle")

demons = {
          "predict_load_demon" : predict_load_demon,
          "predict_zero_on_policy" : predict_zero_on_policy,
          "ten_step_on_policy_load_demon" : ten_step_on_policy_load_demon,
          "ten_step_load_demon" : ten_step_load_demon,
          "ten_step_temp_demon": ten_step_temp_demon,
          "predict_temp_demon": predict_temp_demon,
          "five_step_load_demon" : five_step_load_demon,
          "five_step_temp_demon" :five_step_temp_demon,
          "predict_zero_angle_demon" : predict_zero_angle_demon,
          "return_to_zero_angle_demon" : return_to_zero_angle_demon,
          "return_to_two_angle_demon" : return_to_two_angle_demon,
          }

# for d in range(15):
#     demon = GTDLambda(alpha=0.01, beta=0.001, lambd=0.9, gamma=zero_angle_gamma, cumulant_function=zero_angle_cumulant,
#                                function_approximator=approximator, policy=side_to_side_policy,
#                                track_rupee=True, track_ude=False, name="predict_zero_angle", save_data=False)
#     demons[d] = demon
pavlov_functions = {"prevent_overload" : prevent_overload}

if __name__ == '__main__':
    horde = Horde(observation_class=observations,
                  approximators=approximators,
                  policy_controller=policy_controller, 
                  policies=policies, 
                  demons=demons,
                  plots=plots,)
                  # pavlov_functions,
    horde.run(10000)