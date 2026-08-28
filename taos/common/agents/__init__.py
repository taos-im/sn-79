# SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
# SPDX-License-Identifier: MIT
import os
import bittensor as bt
import time
from pathlib import Path
from abc import ABC, abstractmethod  # Importing the ABC class and abstractmethod decorator for creating abstract base classes
from fastapi import APIRouter
from taos.common.protocol import SimulationStateUpdate, AgentResponse, EventNotification  # Importing required classes for simulation state and agent responses

# Defining an abstract base class for simulation agents
class SimulationAgent(ABC):
    """Abstract base every agent process implements.

    Attributes:
        uid (int): The agent's uid on the subnet, set at launch.
    """
    def __init__(self, uid, config, log_dir = None):
        """
        Initializer method that sets up the agent's unique ID and configuration.
        """
        self.uid = uid
        self.config = config
        if not log_dir:
            log_dir = f"logs/{uid}"
        self.log_dir = log_dir
        default_data_dir = str(Path(__file__).resolve().parents[3] / "agents" / "data")
        self.data_dir = getattr(self.config, 'data_dir', default_data_dir)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.output_dir = os.path.join(self.data_dir, str(self.uid))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.state_file = os.path.join(log_dir, 'state.mp')
        self.router = APIRouter()
        # response_model=None: handle() returns a Pydantic union FinanceAgentResponse | ExchangeAgentResponse,
        # but ExchangeAgentResponse is a sentinel-stub class on the public sim-only surface (the real
        # Pydantic class lives in the excluded taos.im.protocol.exchange). FastAPI rejects sentinel classes
        # as response_models. Disable response-model generation; serialization still happens via Pydantic
        # on the returned object itself.
        self.router.add_api_route("/handle", self.handle, methods=["POST"], response_model=None)
        self.initialize()  # Calling the abstract method to perform any agent-specific setup

    def handle(self, state: SimulationStateUpdate) -> AgentResponse:
        """
        Method to handle a new simulation state update.
        """
        start=time.time()
        self.update(state)  # Update the agent's state based on the new simulation state
        bt.logging.debug(f"Updated ({time.time() - start}s)")
        start=time.time()
        response = self.respond(state)  # Generate a response based on the current state
        bt.logging.debug(f"Responded ({time.time() - start}s)")
        start=time.time()
        self.report(state, response)  # Report the state and response (for logging or other purposes)
        bt.logging.debug(f"Reported ({time.time() - start}s)")
        return response  # Return the generated response

    def process(self, notification: EventNotification) -> EventNotification:
        """
        Method to handle a new event notification.
        """
        notification.acknowledged = True
        return notification

    @abstractmethod
    def initialize(self):
        """
        Abstract method for initialization logic, to be implemented by subclasses.
        """
        ...

    @abstractmethod
    def update(self, state: SimulationStateUpdate) -> None:
        """
        Abstract method to update the agent's state, to be implemented by subclasses.
        
        Args:
            state (taos.common.protocol.SimulationStateUpdate): The synapse object containing the latest simulation state update.

        Returns:
            None
        """
        ...

    @abstractmethod
    def respond(self, state: SimulationStateUpdate) -> AgentResponse:
        """
        Abstract method to create a response based on the current state, to be implemented by subclasses.
        
        Args:
            state (taos.common.protocol.SimulationStateUpdate): The synapse object containing the latest simulation state update.

        Returns:
            taos.common.protocol.AgentResponse: AgentResponse object which will be attached to the synapse for return to querying validator.
        """
        ...

    @abstractmethod
    def report(self, state: SimulationStateUpdate, response: AgentResponse) -> None:
        """
        Abstract method for reporting the state and response, to be implemented by subclasses.
        
        Args:
            state (taos.common.protocol.SimulationStateUpdate): The synapse object containing the latest simulation state update.
            response (taos.common.protocol.AgentResponse): AgentResponse object which will be attached to the synapse for return to querying validator.

        Returns:
        """
        ...

def launch(agent_class):
    """Run an agent class as the miner's strategy process.

    Args:
        agent_class: The agent subclass to instantiate and serve.
    """
    import argparse
    import uvicorn
    from taos.common.config import ParseKwargs
    from fastapi import FastAPI
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument("--agent_id", type=int, required=True)
    parser.add_argument("--params",nargs='*',action=ParseKwargs)
    args = parser.parse_args()
    app = FastAPI()
    agent = agent_class(args.agent_id, args.params)
    app.include_router(agent.router)    
    bt.logging.set_info()
    uvicorn.run(app, port=args.port)