# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import gymnasium as gym
import inspect
import math
import numpy as np
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import MISSING
from typing import Any, ClassVar

import isaacsim.core.utils.torch as torch_utils
import omni.kit.app
import omni.log
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.version import get_version

from isaaclab.managers import EventManager
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.noise import NoiseModel
from isaaclab.utils.timer import Timer

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg



class DirectRLEnv_custom(DirectRLEnv):
    def __init__(self, cfg: DirectRLEnvCfg, render_mode: str | None = None, 
                    rigid_patch_count = 163840,
                    gpu_temp_buffer_capacity = 16777216,
                    gpu_max_rigid_contact_count = 8388608,
                    gpu_heap_capacity = 67108864,
                    gpu_found_lost_pairs_capacity = 2097152,
                    gpu_found_lost_aggregate_pairs_capacity = 33554432,
                    gpu_total_aggregate_pairs_capacity = 2097152,               
                    **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration object for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.

        Raises:
            RuntimeError: If a simulation context already exists. The environment must always create one
                since it configures the simulation context and controls the simulation.
        """

        # check that the config is valid

        cfg.validate()
        # store inputs to class
        self.cfg = cfg
        # store the render mode
        self.render_mode = render_mode
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            omni.log.warn("Seed not set for the environment. The environment creation may not be deterministic.")

        # create a simulation context to control the simulator

        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")
        




        ############## chansol ###############################################################################
        import isaacsim.core.utils.stage as stage_utils
        stage = stage_utils.get_current_stage()
        stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuMaxRigidPatchCount").Set(              rigid_patch_count)
        stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuTempBufferCapacity").Set(              gpu_temp_buffer_capacity)
        stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuMaxRigidContactCount").Set(            gpu_max_rigid_contact_count)
        stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuHeapCapacity").Set(                    gpu_heap_capacity)
        stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuFoundLostPairsCapacity").Set(          gpu_found_lost_pairs_capacity)
        stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuFoundLostAggregatePairsCapacity").Set( gpu_found_lost_aggregate_pairs_capacity)
        stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:gpuTotalAggregatePairsCapacity").Set(     gpu_total_aggregate_pairs_capacity)
        ##########################################################################################








        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step."
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            omni.log.warn(msg)

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(self.cfg.scene)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            # self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
            self.viewport_camera_controller = None
            # pass
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()





                # update scene to pre populate data buffers for assets and sensors.
                # this is needed for the observation manager to get valid tensors for initialization.
                # this shouldn't cause an issue since later on, users do a reset over all the environments so the lazy buffers would be reset.
                self.scene.update(dt=self.physics_dt)

        # -- event manager used for randomization
        if self.cfg.events:
            self.event_manager = EventManager(self.cfg.events, self)
            print("[INFO] Event Manager: ", self.event_manager)

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # check if debug visualization is has been implemented by the environment
        source_code = inspect.getsource(self._set_debug_vis_impl)
        self.has_debug_vis_implementation = "NotImplementedError" not in source_code
        self._debug_vis_handle = None

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # allocate dictionary to store metrics
        self.extras = {}

        # initialize data and constants
        # -- counter for simulation steps
        self._sim_step_counter = 0
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_time_outs = torch.zeros_like(self.reset_terminated)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.sim.device)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # setup noise cfg for adding action and observation noise
        if self.cfg.action_noise_model:
            self._action_noise_model: NoiseModel = self.cfg.action_noise_model.class_type(
                self.cfg.action_noise_model, num_envs=self.num_envs, device=self.device
            )
        if self.cfg.observation_noise_model:
            self._observation_noise_model: NoiseModel = self.cfg.observation_noise_model.class_type(
                self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
            )

        # perform events at the start of the simulation
        if self.cfg.events:
            if "startup" in self.event_manager.available_modes:
                self.event_manager.apply(mode="startup")

        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        # print the environment information
        print("[INFO]: Completed setting up the environment...")
