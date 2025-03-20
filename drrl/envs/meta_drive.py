import copy
import os
from collections import defaultdict
from math import cos, sin
from typing import TYPE_CHECKING, Any, Dict
import logging

import numpy as np
from metadrive import MetaDriveEnv
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.utils import get_np_random, clip
from metadrive import (MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
                       MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv)
from metadrive.component.sensors.rgb_camera import RGBCamera

import gymnasium as gym

class SafetyWrapper(MetaDriveEnv):

    def __init__(self, config: Dict[str, Any]):
        # config["neighbours_distance"] = 40
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
        super().__init__(config)

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def vehicles_dict(self):
        return self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle))

    def reward_function(self, vehicle_id: str):
        original_rewards, infos = super().reward_function(vehicle_id)

        penalty = self.front_end_penalty()

        return original_rewards + penalty, infos

    def front_end_penalty(self):
        direction = self.agents['default_agent'].heading

        observations = self.get_single_observation().observe(self.agents['default_agent'])
        angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        # front end distance
        if angle < 0:
            angle += 360
        front_index = int(angle / (360 / 240))
        front_end_distance = None
        if front_index - 5 < 0:
            front_end_distance = np.min(np.concatenate((observations[-240:][front_index - 5:], observations[-240:][:front_index + 5])))
        elif front_index + 5 > 0:
            front_end_distance = np.min(np.concatenate((observations[-240:][front_index - 5:], observations[-240:][:front_index + 5 - 240])))
        else:
            front_end_distance = np.min(observations[-240:][front_index - 5:front_index + 5])

        # 5m is the safe distance
        front_end_distance_reward = - 0.5 * (1 - front_end_distance) if front_end_distance < 0.1 else 0

        return front_end_distance_reward


def make_ai_vs_bot_env(cfg):
    env = SafetyWrapper(dict(
        traffic_mode="trigger",
        map="C",
        log_level=logging.INFO,
        accident_prob=1.0,
        traffic_density=0.2,
        # use_render=False,
        # agent_observation=LidarStateObservation,
        # image_observation=True,
        # norm_pixel=False,
        # sensors=dict(rgb_camera=(RGBCamera, 512, 256)),
    ))

    return env

class MultiAgentSafetyWrapper(gym.Wrapper):

    def __init__(self, env, cfg):
        super().__init__(env)
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
        self.phi = cfg['phi'] * np.pi # 0 ~ 90 -> 0 ~ 0.5 degree
        self.safe_distance_ratio = cfg['safe_distance_ratio'] # 5m
        self.nei_distance = cfg['neighbours_distance']
        self.right_of_way = cfg['right_of_way']

        # agent str_id -> id to id -> str_id
        self.vehicles_id_to_agent_id = {}
        self.agent_id_to_vehicles_id = {}
        self.prestep_agents = None

    @property
    def vehicles(self):
        return list(self.env.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def vehicles_dicts(self):
        return self.env.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle))


    def step(self, actions):
        """
                Other vehicles' info: [
                                    Projection of distance between ego and another vehicle on ego vehicle's heading direction,
                                    Projection of distance between ego and another vehicle on ego vehicle's side direction,
                                    Projection of speed between ego and another vehicle on ego vehicle's heading direction,
                                    Projection of speed between ego and another vehicle on ego vehicle's side direction,
                                    ] * 4, dim = 16

                Lidar points: 240/70 lidar points surrounding vehicle, starting from the vehicle head in clockwise direction

                :param vehicle: BaseVehicle
                :return: observation in 9 + 10 + 16 + 240/70 dim
        """
        if self.prestep_agents is None:
            self.prestep_agents = self.env.agents

        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        if len(self.vehicles_id_to_agent_id) == 0:
            for k, v in self.env.agents.items():
                self.vehicles_id_to_agent_id[v.id] = k
                self.agent_id_to_vehicles_id[k] = v.id

        vehicles_dicts = self.vehicles_dicts
        # self.update_extra_observations(observations)

        # for vehicle_id, observation in observations.items():
        #     if rewards[vehicle_id] is None:
        #         rewards[vehicle_id] = 0
        #     if rewards[vehicle_id] > 0 and rewards[vehicle_id] < 2:
        #         rewards[vehicle_id] *= 0.1
        # udpate distance
        self.update_nei()

        # update reward
        if self.safe_distance_ratio > 0:
            for vehicle_id, observation in observations.items():
                new_rewards = self.reward_function(vehicle_id, observation)
                rewards[vehicle_id] += new_rewards

        self.update_more_info(observations=observations, infos=infos)

        # right-of-way on responsibility
        crash_agent_ids = []
        for k, v in infos.items():
            if v['crash_vehicle']:
                crash_agent_ids.append(k)

        # which 2 vehicles crash pair
        crash_agent_pairs = []
        has_been_added = set()
        for crash_agent_id in crash_agent_ids:
            # select a min distance crash vehicle from self.nei_distance[crahs_agent_id]
            if crash_agent_id in has_been_added:
                continue

            min_distance = float("inf")
            min_distance_agent_id = None
            for agent_id, distance in self.distance_map[crash_agent_id].items():
                if distance < min_distance and agent_id in crash_agent_ids:
                    min_distance = distance
                    min_distance_agent_id = agent_id

            if min_distance_agent_id is not None:
                crash_agent_pairs.append((crash_agent_id, min_distance_agent_id))
                has_been_added.add(crash_agent_id)
                has_been_added.add(min_distance_agent_id)

        # find responsible vehicle
        if self.right_of_way:
            for crash_agent_id, min_distance_agent_id in crash_agent_pairs:
                v_id = self.agent_id_to_vehicles_id[crash_agent_id]
                other_v_id = self.agent_id_to_vehicles_id[min_distance_agent_id]
                # simple right of way
                # right is all duty
                if vehicles_dicts[v_id].position[0] > vehicles_dicts[other_v_id].position[0]:
                    rewards[crash_agent_id] += 10
                    rewards[min_distance_agent_id] -= 10
                else:
                    rewards[crash_agent_id] -= 10
                    rewards[min_distance_agent_id] += 10

        rewards_with_nei = {}
        # nei reward
        for vehicle_id in rewards.keys():
            nei_reward = 0
            nei_count = 0
            if vehicle_id in self.distance_map:
                for nei_id, distance in self.distance_map[vehicle_id].items():
                    if distance < self.nei_distance and nei_id in rewards:
                        nei_reward += rewards[nei_id]
                        nei_count += 1
            # print(nei_count, "nei_reward: ", nei_reward, " self reward ", rewards[vehicle_id])
            if nei_count > 0:
                rewards_with_nei[vehicle_id] = np.cos(self.phi) * rewards[vehicle_id] + np.sin(self.phi) * nei_reward / nei_count
            else:
                rewards_with_nei[vehicle_id] = np.cos(self.phi) * rewards[vehicle_id]

        rewards = rewards_with_nei

        self.prestep_agents = self.env.agents
        return observations, rewards, terminations, truncations, infos

    def reward_function(self, vehicle_id: str, observation):
        penalty = self.front_end_penalty(vehicle_id, observation)

        return penalty

    def front_end_penalty(self, vehicle_id, observation):
        if vehicle_id not in self.env.agents:
            return 0

        direction = self.env.agents[vehicle_id].heading

        angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        # front end distance
        # TODO only car
        if angle < 0:
            angle += 360
        front_index = int(angle / (360 / 72))
        front_end_distance = None
        if front_index - 5 < 0:
            front_end_distance = np.min(np.concatenate((observation[-72:][front_index - 5:], observation[-72:][:front_index + 5])))
        elif front_index + 5 > 0:
            front_end_distance = np.min(np.concatenate((observation[-72:][front_index - 5:], observation[-72:][:front_index + 5 - 72])))
        else:
            front_end_distance = np.min(observation[-72:][front_index - 5:front_index + 5])

        # 5m is the safe distance
        front_end_distance_reward = - 0.5 * (1 - front_end_distance) if front_end_distance < self.safe_distance_ratio else 0

        return front_end_distance_reward

    def update_more_info(self, observations, infos):
        for vehicle_id, observation in observations.items():
        # def limited_lidar(observation, heading):
            # print(vehicle)
            if vehicle_id not in self.env.agents:
                continue

            cv = self.env.agents[vehicle_id]
            direction = cv.heading
            angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
            # front end distance
            # TODO only car
            if angle < 0:
                angle += 360

            front_index = int(angle / (360 / 72))
            front_end_distance = None
            if front_index - 5 < 0:
                front_end_distance = np.min(np.concatenate((observation[-72:][front_index - 5:], observation[-72:][:front_index + 5])))
            elif front_index + 5 > 0:
                front_end_distance = np.min(np.concatenate((observation[-72:][front_index - 5:], observation[-72:][:front_index + 5 - 72])))
            else:
                front_end_distance = np.min(observation[-72:][front_index - 5:front_index + 5])

            # limit lidar
            limited_lidar = np.count_nonzero(observation[-72:] == 1) / 72 # more is clear

            def in_conflict_zone(position):
                # x: 62 ~ 74[mid] ~ 86
                # y: 17 ~ 5[mid] ~ -7
                if 62 < position[0] < 86 and -7 < position[1] < 17:
                    return True
                return False

            if vehicle_id in infos:
                infos[vehicle_id]['front_end_distance'] = front_end_distance
                infos[vehicle_id]['limited_lidar'] = limited_lidar
                if front_end_distance < 0.5:
                    # too close
                    infos[vehicle_id]['front_end_distance_too_close'] = 1
                else:
                    infos[vehicle_id]['front_end_distance_too_close'] = 0

                if in_conflict_zone(cv.position):
                    infos[vehicle_id]['in_conflict_zone'] = 1
                else:
                    infos[vehicle_id]['in_conflict_zone'] = 0

                if len(self.distance_map[vehicle_id]) > 0:
                    infos[vehicle_id]['pair_distance'] = np.mean(list(self.distance_map[vehicle_id].values()))


    def reset(self, seed = None):
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
        self.vehicles_id_to_agent_id = {}
        self.agent_id_to_vehicles_id = {}
        self.prestep_agents = None
        observations, infos = self.env.reset(seed=seed)
        self.update_more_info(observations=observations, infos=infos)
        # self.update_extra_observations(observations)
        return observations, infos

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    # 计算距离表
    def update_nei(self):
        self.distance_map = defaultdict(lambda: defaultdict(lambda: float("inf")))
        vehicles_dicts = self.vehicles_dicts
        # vehicle_ids = list(self.env.agents.keys())
        vehicle_ids = list(self.vehicles_id_to_agent_id.keys())
        for i in range(len(vehicle_ids)):
            for j in range(i + 1, len(vehicle_ids)):
                vehicle_id_i = vehicle_ids[i]
                vehicle_id_j = vehicle_ids[j]

                agent_id_i = self.vehicles_id_to_agent_id[vehicle_id_i]
                agent_id_j = self.vehicles_id_to_agent_id[vehicle_id_j]

                if vehicle_id_i in vehicles_dicts and vehicle_id_j in vehicles_dicts:
                    position_i = vehicles_dicts[vehicle_id_i].position
                    position_j = vehicles_dicts[vehicle_id_j].position

                    distance = np.sqrt((position_i[0] - position_j[0]) ** 2 + (position_i[1] - position_j[1]) ** 2)

                    self.distance_map[agent_id_i][agent_id_j] = distance
                    self.distance_map[agent_id_j][agent_id_i] = distance

    def update_extra_observations(self, observations):
        vehicles_dicts = self.vehicles_dicts

        # update observations
        for agent_id in observations.keys():
            new_observation = np.zeros(6)
            if agent_id in self.agent_id_to_vehicles_id and self.agent_id_to_vehicles_id[agent_id] in vehicles_dicts:
                vehicle = vehicles_dicts[self.agent_id_to_vehicles_id[agent_id]]
                # bounding_box = np.array(vehicle.bounding_box).reshape(-1)
                position = np.array(vehicle.position)
                local_position = np.array(vehicle.navigation.final_lane.local_coordinates(vehicle.position))
                # heading = np.array(vehicle.heading)
                # pitch = vehicle.pitch
                # roll = vehicle.roll
                speed = vehicle.speed
                steering = vehicle.steering
                # left_side_distance = vehicle.dist_to_left_side
                # right_side_distance = vehicle.dist_to_right_side

                # new_observation = np.concatenate([
                #     bounding_box, position, local_position, heading, [pitch, roll, speed, steering, left_side_distance, right_side_distance],
                # ])

                new_observation = np.concatenate([
                    position, local_position, [speed, steering],
                ])

                # print(vehicle.navigation.get_state(), position, local_position, vehicle.navigation.final_lane.length - local_position[0])

            observations[agent_id] = np.concatenate([new_observation, observations[agent_id]])


def meta_drive_env(cfg):
    env_map = dict(
        roundabout=MultiAgentRoundaboutEnv,
        intersection=MultiAgentIntersectionEnv,
        tollgate=MultiAgentTollgateEnv,
        bottleneck=MultiAgentBottleneckEnv,
        parkinglot=MultiAgentParkingLotEnv,
        pgma=MultiAgentMetaDrive,
    )

    # Omega DictConfig to dict
    env_config = dict(copy.deepcopy(cfg.env))
    env_name = env_config.pop("env_name")

    env = env_map[env_name](
        dict(
            num_agents=cfg.policy.placeholder.max_agents,
            delay_done=5,
            allow_respawn=False,
            start_seed=cfg.seed,
            use_lateral_reward=False,
            # crash_vehicle_cost=1.5,
            # out_of_road_cost=1.5,
            crash_done=False,
            out_of_road_done=False,
            # use_render=False,
            vehicle_config = {
                "enable_reverse": True,
            },
        )
    )

    env = MultiAgentSafetyWrapper(env, env_config)

    return env
