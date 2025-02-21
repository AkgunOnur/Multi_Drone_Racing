import os
import sys
import numpy as np
import pybullet as p
import gymnasium
from gymnasium import spaces
from collections import deque
import time
from PIL import Image
import pybullet_data
import pkg_resources
from datetime import datetime
from scipy.spatial.transform import Rotation
from envs.BaseDrone_SelfPlay import BaseDrone_SelfPlay


from utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from control.DSLPIDControl import DSLPIDControl
from utils.utils import record_video, GateNavigator


gate_folder = "assets/race_gate.urdf"


class MultiGates_v0(BaseDrone_SelfPlay):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 num_dumb_drones: int=1,
                 discrete_action : bool = True,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 record_folder=None,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 max_timesteps = 2000,
                 dumb_no_model = False

                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        self.timesteps = 0
        self.action_size = 6
        self.MAX_TIMESTEPS = max_timesteps
        self.EPISODE_LEN_SEC = 8
        #### Create a buffer for the last .5 sec of actions ########
        self.BUFFER_SIZE = 50#int(ctrl_freq//2)
        self.N_buffer_gate = 2
        self.num_closest_drones = 2
        self.action_buffer = dict()
        self.distance_buffer = dict()
        self.opponent_buffer = dict()
        ####
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.record_folder = record_folder
        self.NUM_DUMB_DRONES = num_dumb_drones
        self.discrete_action = discrete_action
        self.num_actions = 6
        self.dumb_no_model = dumb_no_model

        chn = 0.05
        self.action_mapping = {
            0: [chn, 0, 0],
            1: [-chn, 0, 0],
            2: [0, chn, 0],
            3: [0, -chn, 0],
            4: [0, 0, chn],
            5: [0, 0, -chn]
        }


        self.gate_positions = np.array([
                            [-1.0, 0, 1.0],
                            [-1.2, -1.0, 1.0],
                            [-0.5, -1.5, 1.0],
                            [0.0, -0.75, 1]])
    
        self.gate_rpy = np.array([
                    [0, 0, np.pi/4],
                    [0, 0, np.pi/2],
                    [0, 0, 0],
                    [0, 0, np.pi/2]])
        

        assert len(self.gate_positions) == len( self.gate_rpy)

        self.N_GATES = len(self.gate_positions)
        
        self.gate_quats = [p.getQuaternionFromEuler(euler) for euler in self.gate_rpy]
        self.navigators = [GateNavigator(self.gate_positions) for i in range(num_drones)]


        self.GATE_IDS = []

        dumb_agents = [f"dumb_drone_{n}" for n in range(num_dumb_drones)]
        agents = [f"drone_{n}" for n in range(num_drones)]


        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         num_dumb_drones=num_dumb_drones,
                         agents=agents,
                         dumb_agents=dumb_agents,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         record_folder=record_folder,
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        

        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID, ActionType.POS]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                # self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
                self.ctrl = {agent: DSLPIDControl(drone_model=DroneModel.CF2X) for agent in self.all_agents}
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")


        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)


        
        self.action_space = self.actionSpace()

        self.observation_space = self._observationSpace()



    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """


        for i in range(len(self.gate_positions)):
            self.GATE_IDS.append(p.loadURDF(gate_folder,
                                basePosition=self.gate_positions[i],
                                baseOrientation=self.gate_quats[i],
                                useFixedBase=True,
                                physicsClientId=self.CLIENT))
        

    ################################################################################        
    
    def actionSpace(self, seed=None):
        low = [-0.05, -0.05, -0.05, -0.01]
        high = [0.05, 0.05, 0.05, 0.01]

        if self.discrete_action:
            return spaces.MultiDiscrete([self.action_size] * self.NUM_DRONES)
        else:    
            act_lower_bound = np.array([low for i in range(self.NUM_DRONES)], dtype=np.float32)
            act_upper_bound = np.array([high for i in range(self.NUM_DRONES)], dtype=np.float32)

            return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _observationSpace(self, agent=None):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ

            lo = -np.inf
            hi = np.inf

            drone_lower_bound = [lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo]
            drone_upper_bound = [hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi]

            obs_lower_bound = np.array(drone_lower_bound * 1)
            obs_upper_bound = np.array(drone_upper_bound * 1)
            
            # For action information
            for i in range(self.BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([lo,lo,lo] * 1 )])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([hi,hi,hi] * 1 )])

            # For the closest gate informaion
            for i in range(self.BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([lo,lo] * self.N_buffer_gate)])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([hi,hi] * self.N_buffer_gate)])

            # For the other agents info
            for i in range(self.BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([lo,lo] * self.num_closest_drones)])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([hi,hi] * self.num_closest_drones)])

            # print ("obs bound 2: ", obs_lower_bound.shape)
            
            return spaces.Box(low=np.array([obs_lower_bound] * self.NUM_DRONES), high=np.array([obs_upper_bound] * self.NUM_DRONES), dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")



    ################################################################################

    def _preprocessAction(self,
                          action_list
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """

                
        # targets_to_store = np.zeros((self.NUM_DRONES, 3))
        
        # rpm = np.zeros((self.NUM_DRONES,4))
        rpm = dict()


        # num_drones = len(action_list)
        # action = np.array([action_list[f'drone_{i}'] for i in range(num_drones)])

        # print("action_list: ", action_list)

        # for k in range(self.NUM_DRONES):
        for agent_index, agent in enumerate(self.all_agents):
            state = self._getDroneStateVector(drone_name=agent)      

            self.distance_buffer[agent].append(self._calculateTwoRelativeGateInfo(agent))

            action_k = action_list[agent_index]

            # print ("agent_index:  ", agent_index," ", agent , " action: ", action_k)
            
            # print ("drone ", agent, " action: ", action_k, " ", action_mapping[action_k])
                  
            drone_pos = np.array(state[0:3])
            drone_vel = state[10:13]

            self.navigators[agent].update_drone_position(drone_pos)
            gate_index = self.navigators[agent].current_gate_index
            
            # distance_to_gate = np.linalg.norm(drone_pos - self.gate_positions[gate_index])
            # print (f"Drone: {agent} Gate: {gate_index} Distance: {distance_to_gate:.4f}")

            if self.ACT_TYPE == ActionType.POS:
                
                gate_pos = self.gate_positions[gate_index]
                gate_orient = self.gate_rpy[gate_index]
                target_orient = gate_orient

                if self.discrete_action:
                    target_pos = gate_pos + self.action_mapping[action_k]
                else:
                    target_pos = gate_pos + action_k[0:3]
                    target_orient[2] += action_k[3]

                # Use a PID based planner for dumb agent
                # if agent in self.dumb_agents:
                #     target_pos = gate_pos
                #     target_orient = gate_orient

                if self.dumb_no_model and agent in self.dumb_agents:
                    target_pos = gate_pos

                self.action_buffer[agent].append(target_pos)

                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target_pos,
                    step_size=1,
                )
                
                rpm_k, _, _ = self.ctrl[agent].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos,
                                                        target_rpy=target_orient,
                                                        # target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[agent] = rpm_k

            elif self.ACT_TYPE == ActionType.VEL:

                gate_pos = self.gate_positions[gate_index]
                gate_orient = self.gate_rpy[gate_index]
                target_orient = gate_orient

                if self.discrete_action:
                    target_vel = drone_vel + self.action_mapping[action_k]
                else:
                    target_vel = drone_vel + action_k[0:3]
                    target_orient[2] += action_k[3]

                
                # Use a PID based planner for dumb agent
                # if agent in self.dumb_agents:
                #     target_vel = drone_vel

                if self.dumb_no_model and agent in self.dumb_agents:
                    # print ("agent: ", agent)
                    target_vel = drone_vel
                
                self.action_buffer[agent].append(target_vel)

                

                if np.linalg.norm(target_vel) != 0:
                    v_unit_vector = target_vel/ np.linalg.norm(target_vel)
                else:
                    v_unit_vector = np.zeros(3)


                temp, _, _ = self.ctrl[agent].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=gate_pos,
                                                        target_rpy=target_orient,
                                                        target_vel=self.SPEED_LIMIT * v_unit_vector # target the desired velocity vector
                                                        )
                
                rpm[agent] = temp
            
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()


        # print ("targets_to_store: ", targets_to_store)
        
        return rpm
    
    ################################################################################

    def observe(self, agent:str="drone_1"):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """

        agent_observation = []
        dumb_agent_observation = []
        
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12

            n_obs = 12  # Number of base drone state variables

            # for i in range(self.NUM_DRONES):
            for agent_i in self.all_agents:
                
                opponent_obs = []
                distances = []
                # Get drone state vector
                state_i = self._getDroneStateVector(agent_i)
                drone_i_pos = np.array(state_i[0:3])
                drone_i_quat = np.array(state_i[3:7])
                drone_i_rot = Rotation.from_quat(drone_i_quat)

                for agent_j in self.all_agents:
                    if agent_i == agent_j:
                        continue
                    
                    state_j = self._getDroneStateVector(agent_j)
                    drone_j_pos = np.array(state_j[0:3])
                    drone_j_quat = np.array(state_j[3:7])
                    drone_j_rot = Rotation.from_quat(drone_j_quat)

                    relative_rot = drone_i_rot.inv() * drone_j_rot
                    relative_euler = relative_rot.as_euler('xyz', degrees=False)

                    # opponent_obs.append([np.linalg.norm(drone_j_pos - drone_i_pos),np.linalg.norm(relative_euler)])

                    distance = np.linalg.norm(drone_j_pos - drone_i_pos)
                    observation = [distance, np.linalg.norm(relative_euler)]

                    distances.append(distance)
                    opponent_obs.append(observation)
                    

                # Sort the distances and observations together based on distances
                sorted_indices = np.argsort(distances)
                sorted_distances = [distances[i] for i in sorted_indices]
                sorted_opponent_obs = [opponent_obs[i] for i in sorted_indices]

                # Take the observations of the closest drones
                closest_opponent_obs = sorted_opponent_obs[:self.num_closest_drones]

                # If there is only one opponent drone, duplicate its observation
                for i in range(len(closest_opponent_obs), self.num_closest_drones):
                    closest_opponent_obs.append(closest_opponent_obs[-1])

                # Store the closest opponent observations for the current agent
                # agent_i.closest_opponent_obs = closest_opponent_obs


                self.opponent_buffer[agent_i].append(np.array(closest_opponent_obs).flatten())


                # Extract relevant state variables
                drone_state = np.hstack([state_i[0:3], state_i[7:10], state_i[10:13], state_i[13:16]]).reshape(n_obs,)

                # Get action buffer for the current drone
                act_buf = np.array(self.action_buffer[agent_i])
                
                # Get distance buffer for the current drone
                dist_buf = np.array(self.distance_buffer[agent_i])

                opp_buffer = np.array(self.opponent_buffer[agent_i])
                
                # Concatenate drone state, action buffer, and distance buffer
                drone_obs = np.concatenate((drone_state, act_buf.flatten(), dist_buf.flatten(), opp_buffer.flatten()))
                
                # Store observation for the current drone in the dictionary

                if agent_i in self.agents:
                    agent_observation.append(drone_obs.astype('float32'))
                elif agent_i in self.dumb_agents:
                    dumb_agent_observation.append(drone_obs.astype('float32'))


            # print ("obs size: ", obs.shape)
            return np.array(agent_observation), np.array(dumb_agent_observation)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary.observe()")

    def _calculateRelativGateInfoAll(self):
        gate_info = []
        for agent in range(self.all_agents):
            state = self._getDroneStateVector(drone_name=agent)
            distances_k = np.zeros((self.N_GATES, 2))

            drone_pos = state[0:3]
            drone_quat = state[3:7]

            for i in range(self.N_GATES):
                gate_pos = self.gate_positions[i]
                gate_quat = self.gate_quats[i]

                drone_rot = Rotation.from_quat(drone_quat)
                gate_rot = Rotation.from_quat(gate_quat)

                relative_rot = drone_rot.inv() * gate_rot
                relative_euler = relative_rot.as_euler('xyz', degrees=True)

                distances_k[i, 0] = np.linalg.norm(drone_pos - gate_pos) 
                distances_k[i, 1] = np.linalg.norm(relative_euler)
            
            gate_info.append(distances_k.ravel())

        gate_info = np.array(gate_info)
        # print ("gate info: ", gate_info.shape)

        return gate_info
    
    def _calculateRelativGateInfo(self, agent):
        distances = np.zeros(2)
        # for k in range(self.NUM_DRONES):

        state = self._getDroneStateVector(drone_name=agent)
        i = self.navigators[agent].current_gate_index # closest gate index
    
        drone_pos = np.array(state[0:3])
        drone_quat = np.array(state[3:7])

        gate_pos = self.gate_positions[i]
        gate_quat = self.gate_quats[i]

        drone_rot = Rotation.from_quat(drone_quat)
        gate_rot = Rotation.from_quat(gate_quat)

        relative_rot = drone_rot.inv() * gate_rot
        relative_euler = relative_rot.as_euler('xyz', degrees=True)

        distances[0] = np.linalg.norm(drone_pos - gate_pos) 
        distances[1] = np.linalg.norm(relative_euler)
        
        return distances
    
    def _calculateTwoRelativeGateInfo(self, agent):
        distances = np.zeros(2*self.N_buffer_gate)

        state = self._getDroneStateVector(drone_name=agent)
        index = self.navigators[agent].current_gate_index # closest gate index

        drone_pos = np.array(state[0:3])
        drone_quat = np.array(state[3:7])
        drone_rot = Rotation.from_quat(drone_quat)

        for i in range(index, index + self.N_buffer_gate):
            j = i % self.N_GATES
            gate_pos = self.gate_positions[j]
            gate_quat = self.gate_quats[j]
            gate_rot = Rotation.from_quat(gate_quat)

            relative_rot = drone_rot.inv() * gate_rot
            relative_euler = relative_rot.as_euler('xyz', degrees=True)

            distances[2*(i - index)] = np.linalg.norm(drone_pos - gate_pos) 
            distances[2*(i - index) + 1] = np.linalg.norm(relative_euler)
        
        return distances

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `observe()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        # TODO : initialize random number generator with seed

        original_init_pos = [0,0,0]
        self.GATE_IDS = []
        initial_xyzs = initialize_drones(self.all_agents, original_init_pos)

        self.infos = {agent: {"lap_time": []} for agent in self.all_agents}
        self.n_completion = {agent: 0 for agent in self.all_agents}
        self.previous_lap_time = {agent: 0 for agent in self.all_agents}

        self.prev_gate_index = dict()
        for agent in self.all_agents:
            self.prev_gate_index[agent] = 0
            self.action_buffer[agent] = deque(maxlen=self.BUFFER_SIZE)
            self.distance_buffer[agent] = deque(maxlen=self.BUFFER_SIZE)
            self.opponent_buffer[agent] = deque(maxlen=self.BUFFER_SIZE)

            for i in range(self.BUFFER_SIZE):
                self.action_buffer[agent].append(np.zeros(3))
                self.distance_buffer[agent].append(np.zeros(2*self.N_buffer_gate))
                self.opponent_buffer[agent].append(np.zeros(2*self.num_closest_drones))


        self.timesteps = 0
        self.navigators = {agent: GateNavigator(self.gate_positions) for agent in self.all_agents}

        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping(initial_xyzs)
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        agent_state, dumb_agent_state = self.observe()
        initial_info = self._computeInfo()
        return agent_state, dumb_agent_state, initial_info
    

    def check_gate_collision(self, drone_name):
        for i in range(len(self.gate_positions)):
            gate_collision = p.getContactPoints(bodyA=self.DRONE_IDS[drone_name],
                                            bodyB=self.GATE_IDS[i],
                                            physicsClientId=self.CLIENT)
            if gate_collision:
                # print ("collision ", drone_name, " time: ", self.timesteps)
                return True
        
        return False

    def _computeDroneFailed(self, drone_name):
        distance_threshold = 5

        state = self._getDroneStateVector(drone_name)
        drone_pos = np.array(state[0:3])
        drone_rpy = np.array(state[7:10])

        if np.abs(drone_rpy[0]) > 3*np.pi/4 or np.abs(drone_rpy[1]) > 3*np.pi/4:
            return True
        
        for j in range(len(self.gate_positions)):
            euclidean_dist = np.linalg.norm(drone_pos - self.gate_positions[j])
            if euclidean_dist >= distance_threshold:
                return True

        return False

    

    def _computeTerminated(self):
        """Computes the current terminated value(s) for each drone.

        Returns:
            dict: A dictionary containing the terminated value for each drone, with keys "drone_0", "drone_1", etc.
        """
        terminated = dict()

        for drone_index in range(self.NUM_DRONES):
            terminated[f"drone_{drone_index}"] = False

            if self._computeDroneFailed(drone_index) or self.timesteps >= self.MAX_TIMESTEPS:
                terminated[f"drone_{drone_index}"] = True

            # for drone_j in range(self.NUM_DRONES):
            #     if drone_index == drone_j:
            #         continue

            #     drone_collision = p.getContactPoints(bodyA=self.DRONE_IDS[drone_index],
            #                                         bodyB=self.DRONE_IDS[drone_j],
            #                                         physicsClientId=self.CLIENT)
            #     if drone_collision:
            #         terminated[f"drone_{drone_index}"] = True
            #         break

        return terminated

    def _computeTruncated(self):
        """Computes the current truncated value(s) for each drone.

        Returns:
            dict: A dictionary containing the truncated value for each drone, with keys "drone_0", "drone_1", etc.
        """
        truncated = dict()

        for drone_index in range(self.NUM_DRONES):
            truncated[f"drone_{drone_index}"] = False

        return truncated

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        # info = dict()
        return {}
    


    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `observe()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is over, check the specific implementation of `_computeTerminated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is truncated, check the specific implementation of `_computeTruncated()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current episode is trunacted, always false.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """

        self.timesteps += 1

        #### Save PNG video frames if RECORD=True and GUI=False ####
        if self.RECORD and not self.GUI and self.step_counter%self.CAPTURE_FREQ == 0:
            [w, h, rgb, dep, seg] = p.getCameraImage(width=self.VID_WIDTH,
                                                     height=self.VID_HEIGHT,
                                                     shadow=1,
                                                     viewMatrix=self.CAM_VIEW,
                                                     projectionMatrix=self.CAM_PRO,
                                                     renderer=p.ER_TINY_RENDERER,
                                                     flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                                     physicsClientId=self.CLIENT
                                                     )
            (Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA')).save(os.path.join(self.IMG_PATH, "frame_"+str(self.FRAME_NUM)+".png"))
            #### Save the depth or segmentation view instead #######
            # dep = ((dep-np.min(dep)) * 255 / (np.max(dep)-np.min(dep))).astype('uint8')
            # (Image.fromarray(np.reshape(dep, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            # seg = ((seg-np.min(seg)) * 255 / (np.max(seg)-np.min(seg))).astype('uint8')
            # (Image.fromarray(np.reshape(seg, (h, w)))).save(self.IMG_PATH+"frame_"+str(self.FRAME_NUM)+".png")
            self.FRAME_NUM += 1
            if self.VISION_ATTR:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
                    #### Printing observation to PNG frames example ############
                    self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
                                    img_input=self.rgb[i],
                                    path=self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/",
                                    frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                    )
        #### Read the GUI's input parameters #######################
        if self.GUI and self.USER_DEBUG:
            current_input_switch = p.readUserDebugParameter(self.INPUT_SWITCH, physicsClientId=self.CLIENT)
            if current_input_switch > self.last_input_switch:
                self.last_input_switch = current_input_switch
                self.USE_GUI_RPM = True if self.USE_GUI_RPM == False else False
        if self.USE_GUI_RPM:
            for i in range(4):
                self.gui_input[i] = p.readUserDebugParameter(int(self.SLIDERS[i]), physicsClientId=self.CLIENT)
            clipped_action = np.tile(self.gui_input, (self.NUM_DRONES, 1))
            if self.step_counter%(self.PYB_FREQ/2) == 0:
                self.GUI_INPUT_TEXT = [p.addUserDebugText("Using GUI RPM",
                                                          textPosition=[0, 0, 0],
                                                          textColorRGB=[1, 0, 0],
                                                          lifeTime=1,
                                                          textSize=2,
                                                          parentObjectUniqueId=self.DRONE_IDS[i],
                                                          parentLinkIndex=-1,
                                                          replaceItemUniqueId=int(self.GUI_INPUT_TEXT[i]),
                                                          physicsClientId=self.CLIENT
                                                          ) for i in range(self.NUM_DRONES)]
        #### Save, preprocess, and clip the action to the max. RPM #
        else:
            # clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
            clipped_action = self._preprocessAction(action)
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.PYB_STEPS_PER_CTRL):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            if self.PYB_STEPS_PER_CTRL > 1 and self.PHYSICS in [Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                self._updateAndStoreKinematicInformation()
            #### Step the simulation using the desired physics update ##
            
            for agent in self.all_agents:
                if self.PHYSICS == Physics.PYB:
                    self._physics(clipped_action[agent], agent)
                elif self.PHYSICS == Physics.DYN:
                    self._dynamics(clipped_action[agent], agent)
                elif self.PHYSICS == Physics.PYB_GND:
                    self._physics(clipped_action[agent], agent)
                    self._groundEffect(clipped_action[agent], agent)
                elif self.PHYSICS == Physics.PYB_DRAG:
                    self._physics(clipped_action[agent], agent)
                    self._drag(self.last_clipped_action[agent], agent)
                elif self.PHYSICS == Physics.PYB_DW:
                    self._physics(clipped_action[agent], agent)
                    self._downwash(agent)
                elif self.PHYSICS == Physics.PYB_GND_DRAG_DW:
                    self._physics(clipped_action[agent], agent)
                    self._groundEffect(clipped_action[agent], agent)
                    self._drag(self.last_clipped_action[agent], agent)
                    self._downwash(agent)
            #### PyBullet computes the new state, unless Physics.DYN ###
            if self.PHYSICS != Physics.DYN:
                p.stepSimulation(physicsClientId=self.CLIENT)
            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################

        agent_state, dumb_agent_state = self.observe()
        terminated = {agent: False for agent in self.agents}
        truncated = False
        reward = {agent: 0.0 for agent in self.agents}
        dumb_reward = {agent: 0.0 for agent in self.dumb_agents}
        info = {agent: {} for agent in self.agents}
        done = False

        for agent in self.all_agents:
            if agent in self.agents:
                reward[agent] = 0
                terminated[agent] = False
            else:
                dumb_reward[agent] = 0

            gate_index = self.navigators[agent].current_gate_index

            if self.n_completion[agent] < self.navigators[agent].completed_laps:
                completion_lap_time = self.timesteps - self.previous_lap_time[agent]
                reward_value = 3000 / completion_lap_time
                if agent in self.agents:
                    reward[agent] = reward_value
                else:
                    dumb_reward[agent] = reward_value
                self.n_completion[agent] = self.navigators[agent].completed_laps
                self.previous_lap_time[agent] = self.timesteps
                self.infos[agent]["lap_time"].append(completion_lap_time / self.PYB_FREQ)
                # if agent in self.agents:
                #     print("agent: ", agent, " step:", self.timesteps, "completion:", completion_lap_time, "reward:", reward[agent])
                # else:
                #     print("agent: ", agent, " step:", self.timesteps, "completion:", completion_lap_time, "reward:", dumb_reward[agent])

                    
            # Add a reward for passing through the gate
            elif gate_index > self.prev_gate_index[agent]:
                if agent in self.agents:
                    reward[agent] = 0.5
                else:
                    dumb_reward[agent] = 0.5
                self.prev_gate_index[agent] = np.copy(gate_index)

            elif self._computeDroneFailed(agent):
                if agent in self.agents:
                    reward[agent] = -1.0
                    terminated[agent] = True
                else:
                    dumb_reward[agent] = -1.0

            elif self.check_gate_collision(agent):
                if agent in self.agents:
                    reward[agent] = -0.1
                else:
                    dumb_reward[agent] = -0.1

            if agent in self.agents and self.timesteps >= self.MAX_TIMESTEPS:
                truncated = True
        

        if all(terminated.values()):
            done = True


        # print (f"drone_pos: {drone_pos[0]:.3f} {drone_pos[1]:.3f} {drone_pos[2]:.3f}")
        # print ("gate_index: ", gate_index, " position: ", self.gate_positions[gate_index])

        reward_val = sum(reward.values())
        dumb_reward_val = sum(dumb_reward.values())
        # print ("hooop")

        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.PYB_STEPS_PER_CTRL)
        return agent_state, dumb_agent_state, reward_val, dumb_reward_val, done, truncated, self.infos
    

    
def initialize_drones(all_agents, original_init_pos, margin=0.5, altitude=1.0, max_offset=0.15):
    # np.random.seed(int(time.time()))

    drone_positions = {}
    num_drones = len(all_agents)

    # print ("num_drones: ", num_drones)
    # Calculate the angle between each drone on the circle
    angle_step = 2 * np.pi / num_drones
    
    # Calculate the radius of the circle based on the margin between drones
    radius = margin * num_drones / (2 * np.pi)
    
    for i, agent in enumerate(all_agents):
        # Calculate the angle for the current drone
        angle = i * angle_step
        
        # Calculate the x and y coordinates of the drone on the circle
        x = original_init_pos[0] + radius * np.cos(angle)
        y = original_init_pos[1] + radius * np.sin(angle)
        
        # Add a small random offset to the x and y coordinates
        x += np.random.uniform(-max_offset, max_offset)
        y += np.random.uniform(-max_offset, max_offset)
        
        # Set the z coordinate (altitude) of the drone
        z = np.random.uniform(0, altitude)
        
        # Append the drone's position to the list
        drone_positions[agent] = [x, y, z]
    
    return drone_positions
