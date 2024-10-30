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

"""This module contains the shared state for the abci skill of PeaqAbciApp."""

from typing import Any

from packages.valory.skills.abstract_round_abci.models import BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)
from packages.keyko.skills.peaq_abci.rounds import PeaqAbciApp


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = PeaqAbciApp


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool

class Params(BaseParams):
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        self.number_of_data_points = kwargs.get("number_of_data_points", None)
        self.combinder_api_url = kwargs.get("combinder_api_url", None)
        self.model_api_url = kwargs.get("model_api_url", None)
        self.combinder_api_key = kwargs.get("combinder_api_key", None)
        self.model_id = kwargs.get("model_id", 6)
        self.model_api_key = kwargs.get("model_api_key", None)
        self.temperature = kwargs.get("temperature", None)
        self.max_tokens = kwargs.get("max_tokens", None)
        self.solar_device_id = kwargs.get("solar_device_id", None)
        self.ac_device_id = kwargs.get("ac_device_id", None)

        super().__init__(*args, **kwargs)

