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

"""This package contains round behaviours of PeaqAbciApp."""

from abc import ABC
from typing import Generator, Set, Type, cast
import json

from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)

from packages.keyko.skills.peaq_abci.models import Params
from packages.keyko.skills.peaq_abci.rounds import (
    SynchronizedData,
    PeaqAbciApp,
    CollectDataRound,
    DeviceInteractionRound,
    QueryModelRound,
    RegistrationRound,
    ResetAndPauseRound,
    Event
)
from packages.keyko.skills.peaq_abci.rounds import (
    CollectDataPayload,
    DeviceInteractionPayload,
    QueryModelPayload,
    RegistrationPayload,
    ResetAndPausePayload,
)


class PeaqBaseBehaviour(BaseBehaviour, ABC):
    """Base behaviour for the peaq_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)


class RegistrationBehaviour(PeaqBaseBehaviour):
    """RegistrationBehaviour"""

    matching_round: Type[AbstractRound] = RegistrationRound

    # TODO: implement logic required to set payload content for synchronization
    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        # Step 0: Mock prosumer data

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            energy_data = yield from self.get_combinder_past_data()
            sender = self.context.agent_address
            payload = RegistrationPayload(sender=sender, prosumer_data=energy_data)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_combinder_past_data(self) -> Generator[None, None, str]:
        """
        Get the data from Combinder API which contains the power production and consumption data.
        """
        url = self.params.combinder_api_url + f"/past-data"
        response = yield from self.get_http_response(
            method="GET",
            url=url
        )
        if response.status_code != 200:
            self.context.logger.error(
                f"APICheckBehaviour: url: {url}"
                f"APICheckBehaviour: Could not retrieve data from Combinder API. "
                f"APICheckBehaviour: Received status code {response.status_code}."
            )
            self.context.logger.error(f"APICheckBehaviour: Response body: {response.body}")
            self.context.logger.error(f"APICheckBehaviour: Response headers: {response.headers}")
            return ""
        
        data = json.loads(response.body)
        
        return data

class CollectDataBehaviour(PeaqBaseBehaviour):
    """CollectDataBehaviour"""

    matching_round: Type[AbstractRound] = CollectDataRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            energy_data = yield from self.get_combinder_data()
            
            sender = self.context.agent_address
            payload = CollectDataPayload(sender=sender, prosumer_data=energy_data)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()
    
    def get_combinder_data(self) -> Generator[None, None, str]:
        """
        Get the data from Combinder API which contains the power production and consumption data.
        """
        url = self.params.combinder_api_url + f"/energy-data"
        response = yield from self.get_http_response(
            method="GET",
            url=url
        )
        if response.status_code != 200:
            self.context.logger.error(
                f"APICheckBehaviour: url: {url}"
                f"APICheckBehaviour: Could not retrieve data from Combinder API. "
                f"APICheckBehaviour: Received status code {response.status_code}."
            )
            self.context.logger.error(f"APICheckBehaviour: Response body: {response.body}")
            self.context.logger.error(f"APICheckBehaviour: Response headers: {response.headers}")
            return ""
        
        data = json.loads(response.body)
        
        return {
            "grid_import": data["grid_import"],
            "pv": data["pv"],
            "cet_cest_timestamp": data["cet_cest_timestamp"]
        }

class QueryModelBehaviour(PeaqBaseBehaviour):
    """QueryModelBehaviour"""

    matching_round: Type[AbstractRound] = QueryModelRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""
        # Step 2: Query the model to get the prediction

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            model_prediction = yield from self.query_model()
            sender = self.context.agent_address
            payload = QueryModelPayload(sender=sender, prediction_class=model_prediction)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def query_model(self):
        # Query the model with synchronized data
        url = self.params.model_api_url + f"/predict"
        http_params = {
            "data": self.synchronized_data.prosumer_data
        }
        response = yield from self.get_http_response(
            method="POST",
            url=url,
            content=json.dumps(http_params).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            self.context.logger.error(
                f"QueryModelBehaviour: url: {url}"
                f"QueryModelBehaviour: Could not predict data from Combinder API. "
                f"QueryModelBehaviour: Received status code {response.status_code}."
            )
            self.context.logger.error(f"QueryModelBehaviour: Response body: {response.body}")
            self.context.logger.error(f"QueryModelBehaviour: Response headers: {response.headers}")
            return -1

        data = json.loads(response.body)

        return data["prediction_class"]

class DeviceInteractionBehaviour(PeaqBaseBehaviour):
    """DeviceInteractionBehaviour"""

    matching_round: Type[AbstractRound] = DeviceInteractionRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            device_action = bool(self.synchronized_data.last_prediction_class)
            ret = yield from self.control_device(self.params.device_id, device_action)
            sender = self.context.agent_address
            payload = DeviceInteractionPayload(sender=sender, success=ret["success"], message=ret["message"])

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()
        
    def control_device(self, device_id, device_action):
        # Control the device based on the prediction
        url = self.params.combinder_api_url + f"/device/control"
        response = yield from self.get_http_response(
            method="POST",
            url=url,
            content=json.dumps({
                "device_id": device_id,
                "action": int(device_action)
            }).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        if response.status_code != 200:
            self.context.logger.error(
                f"DeviceInteractionBehaviour: url: {url}\n"
                f"DeviceInteractionBehaviour: Could not control device from Combinder API.\n"
                f"DeviceInteractionBehaviour: Received status code {response.status_code}.\n"
            )
            self.context.logger.error(f"DeviceInteractionBehaviour: Response body: {response.body}")
            self.context.logger.error(f"DeviceInteractionBehaviour: Response headers: {response.headers}")
            return {
                "success": False,
                "message": f"Could not control device. (Status code: {response.status_code})"
            }
        data = json.loads(response.body)

        return data


class ResetAndPauseBehaviour(PeaqBaseBehaviour):
    """ResetAndPauseBehaviour"""

    matching_round: Type[AbstractRound] = ResetAndPauseRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            yield from self.sleep(self.params.reset_pause_duration)
            sender = self.context.agent_address
            payload = ResetAndPausePayload(sender=sender)

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

class PeaqRoundBehaviour(AbstractRoundBehaviour):
    """PeaqRoundBehaviour"""

    initial_behaviour_cls = RegistrationBehaviour
    abci_app_cls = PeaqAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [
        CollectDataBehaviour,
        DeviceInteractionBehaviour,
        QueryModelBehaviour,
        RegistrationBehaviour,
        ResetAndPauseBehaviour
    ]
