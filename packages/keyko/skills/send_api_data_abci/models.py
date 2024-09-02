# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This module contains the shared state for the abci skill of SendAPIDataAbciApp."""

from dataclasses import dataclass
from typing import Any, Dict, List

from packages.valory.skills.abstract_round_abci.models import BaseParams, TypeCheckMixin
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.valory.skills.send_api_data_abci.rounds import SendAPIDataAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = SendAPIDataAbciApp


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool

    
@dataclass(frozen=True)
class FSM(TypeCheckMixin):
    """A dataclass to store information about an FSM."""

    name: str
    description: str
    slug: str

    def to_json(self) -> Dict[str, str]:
        """Get an FSM instance as a json dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "slug": self.slug,
        }


@dataclass(frozen=True)
class FSMSuite(TypeCheckMixin):
    """A dataclass to store information about an FSM suite."""

    name: str
    description: str
    slug: str
    fsms: List[FSM]

    def to_json(self) -> Dict[str, any]:
        """Get an FSMSuite instance as a json dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "slug": self.slug,
            "fsms": [fsm.to_json() for fsm in self.fsms],
        }


@dataclass(frozen=True)
class Instance(TypeCheckMixin):
    """A dataclass to store information about an instance."""

    name: str
    slug: str

    def to_json(self) -> Dict[str, str]:
        """Get an Instance instance as a json dictionary."""
        return {
            "name": self.name,
            "slug": self.slug,
        }

@dataclass(frozen=True)
class Project(TypeCheckMixin):
    """A dataclass to store information about a project."""

    name: str
    description: str
    slug: str

    def to_json(self) -> Dict[str, str]:
        """Get a Project instance as a json dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "slug": self.slug,
        }

@dataclass(frozen=True)
class ProjectDBData(TypeCheckMixin):
    """A dataclass to store project DB data."""

    project: Project
    fsm_suite: FSMSuite
    instance: Instance

    @classmethod
    def from_json_dict(cls, json_dict: dict) -> "ProjectDBData":
        """Get a ProjectDBData instance from a json dictionary."""
        fsms = [FSM(**fsm_dict) for fsm_dict in json_dict["fsm_suite"]["fsms"]]
        fsm_suite = FSMSuite(
            json_dict["fsm_suite"]["name"],
            json_dict["fsm_suite"]["description"],
            json_dict["fsm_suite"]["slug"],
            fsms,
        )
        instance = Instance(**json_dict["instance"])
        project = Project(**json_dict["project"])
        return cls(
            project,
            fsm_suite,
            instance,
        )

    def to_json(self) -> Dict[str, any]:
        """Get a ProjectDBData instance as a json dictionary."""
        return {
            "project": self.project.to_json(),
            "fsm_suite": self.fsm_suite.to_json(),
            "instance": self.instance.to_json(),
        }
        
        
class SendAPIDataParams(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        self.project_db_data: ProjectDBData = ProjectDBData.from_json_dict(
            self._ensure("project_db_data", kwargs, dict)
        )
        self.agent_db_name: str = self._ensure("agent_db_name", kwargs, str)
        self.agent_db_description: str = self._ensure("agent_db_description", kwargs, str)
        self.agent_db_slug: str = self._ensure("agent_db_slug", kwargs, str)
        self.olas_api_x_api_key = self._ensure("olas_api_x_api_key", kwargs, str)
        self.olas_api_url_project_data = self._ensure("olas_api_url_project_data", kwargs, str)
        self.olas_api_url_agent_data = self._ensure("olas_api_url_agent_data", kwargs, str)
        
        # self.coingecko_api_key = kwargs.get("coingecko_api_key", None)
        # self.erc20_token_address = self._ensure("erc20_token_address", kwargs, str)
        # self.fxstreet_api_url = self._ensure("fxstreet_api_url", kwargs, str)
        # self.usbls_statement_page = self._ensure(
        #     "usbls_statement_page", kwargs, str
        # )

        # self.wxdai_contract_address = self._ensure("wxdai_contract_address", kwargs, str)
        # self.wbtc_contract_address = self._ensure("wbtc_contract_address", kwargs, str)

        # self.trade_amount = self._ensure("trade_amount", kwargs, int)
        # self.uniswap_router_address = self._ensure(
        #     "uniswap_router_address", kwargs, str
        # )
        # self.slippage_tolerance = self._ensure("slippage_tolerance", kwargs, float)

        # self.deposit_tracker_address = self._ensure(
        #     "deposit_tracker_address", kwargs, str
        # )
        # self.multisend_contract_address = self._ensure(
        #     "multisend_contract_address", kwargs, str
        # )

        super().__init__(*args, **kwargs)
